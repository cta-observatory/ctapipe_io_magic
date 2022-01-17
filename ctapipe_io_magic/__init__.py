"""
# Event source for MAGIC calibrated data files.
# Requires uproot package (https://github.com/scikit-hep/uproot).
"""

import re
import uproot
import logging
import scipy
import scipy.interpolate
import numpy as np
from pkg_resources import resource_filename
from decimal import Decimal
from astropy.coordinates import Angle
from astropy import units as u
from astropy.time import Time

from ctapipe.io.eventsource import EventSource
from ctapipe.io.datalevels import DataLevel
from ctapipe.core import Container, Field
from ctapipe.core import Provenance
from ctapipe.core.traits import Bool
from ctapipe.coordinates import CameraFrame

from ctapipe.containers import (
    EventType,
    ArrayEventContainer,
    SimulatedEventContainer,
    SimulatedShowerContainer,
    SimulationConfigContainer,
    PointingContainer,
    TelescopePointingContainer,
    TelescopeTriggerContainer,
    MonitoringCameraContainer,
    PedestalContainer,
)

from ctapipe.instrument import (
    TelescopeDescription,
    SubarrayDescription,
    OpticsDescription,
    CameraDescription,
    CameraGeometry,
    CameraReadout,
)

from .mars_datalevels import MARSDataLevel

from .version import __version__

from .constants import (
    MC_STEREO_TRIGGER_PATTERN,
    PEDESTAL_TRIGGER_PATTERN,
    DATA_STEREO_TRIGGER_PATTERN
)

__all__ = ['MAGICEventSource', '__version__']

LOGGER = logging.getLogger(__name__)

degrees_per_hour = 15.0
seconds_per_hour = 3600.

msec2sec = 1e-3
nsec2sec = 1e-9

MAGIC_TO_CTA_EVENT_TYPE = {
    MC_STEREO_TRIGGER_PATTERN: EventType.SUBARRAY,
    PEDESTAL_TRIGGER_PATTERN: EventType.SKY_PEDESTAL,
    DATA_STEREO_TRIGGER_PATTERN: EventType.SUBARRAY,
}

OPTICS = OpticsDescription(
    'MAGIC',
    num_mirrors=1,
    equivalent_focal_length=u.Quantity(16.97, u.m),
    mirror_area=u.Quantity(239.0, u.m**2),
    num_mirror_tiles=964,
)


def load_camera_geometry():
    ''' Load camera geometry from bundled resources of this repo '''
    f = resource_filename(
        'ctapipe_io_magic', 'resources/MAGICCam.camgeom.fits.gz'
    )
    Provenance().add_input_file(f, role="CameraGeometry")
    return CameraGeometry.from_table(f)


class MissingDriveReportError(Exception):
    """
    Exception raised when a subrun does not have drive reports.
    """

    def __init__(self, message):
        self.message = message


class MAGICEventSource(EventSource):
    """
    EventSource for MAGIC calibrated data.

    This class operates with the MAGIC data subrun-wise for calibrated data.

    Attributes
    ----------
    current_run : MarsCalibratedRun
        Object containing the info needed to fill the ctapipe Containers
    datalevel : DataLevel
        Data level according to the definition in ctapipe
    file_ : uproot.ReadOnlyFile
        A ROOT file opened with uproot
    is_mc : bool
        Flag indicating real or simulated data
    mars_datalevel : int
        Data level according to MARS convention
    metadata : dict
        Dictionary containing metadata
    run_numbers : int
        Run number of the file
    simulation_config : SimulationConfigContainer
        Container filled with the information about the simulation
    telescope : int
        The number of the telescope
    use_pedestals : bool
        Flag indicating if pedestal events should be returned by the generator
    """

    use_pedestals = Bool(
           default_value=False,
           help=(
               'If true, extract pedestal evens instead of cosmic events.'
           ),
    ).tag(config=False)

    def __init__(self, input_url=None, config=None, parent=None, **kwargs):
        """
        Constructor

        Parameters
        ----------
        config: traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        parent : ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs: dict
            Additional parameters to be passed.
            NOTE: The file mask of the data to read can be passed with
            the 'input_url' parameter.
        """

        super().__init__(input_url=input_url, config=config, parent=parent, **kwargs)

        # Retrieving the list of run numbers corresponding to the data files
        self.file_ = uproot.open(self.input_url.expanduser())
        run_info = self.parse_run_info()

        self.run_numbers = run_info[0]
        self.is_mc = run_info[1]
        self.telescope = run_info[2]
        self.mars_datalevel = run_info[3]

        self.metadata = self.parse_metadata_info()

        # Retrieving the data level (so far HARDCODED Sorcerer)
        self.datalevel = DataLevel.DL0

        if self.is_mc:
            self.simulation_config = self.parse_simulation_header()

        if not self.is_mc:
            self.is_stereo, self.is_sumt = self.parse_data_info()

        # # Setting up the current run with the first run present in the data
        # self.current_run = self._set_active_run(run_number=0)
        self.current_run = None

        self._subarray_info = self.prepare_subarray_info()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Releases resources (e.g. open files).

        Parameters
        ----------
        exc_type : Exception
            Class of the exception
        exc_val : BaseException
            Type of the exception
        exc_tb : TracebackType
            The traceback
        """

        self.close()

    def close(self):
        """
        Closes open ROOT file.
        """

        self.file_.close()

    @staticmethod
    def is_compatible(file_path):
        """
        This method checks if the specified file mask corresponds
        to MAGIC data files. The result will be True only if all
        the files are of ROOT format and contain an 'Events' tree.

        Parameters
        ----------
        file_path: str
            Path to file

        Returns
        -------
        bool:
            True if the masked files are MAGIC data runs, False otherwise.

        """

        is_magic_root_file = True

        try:
            with uproot.open(file_path) as input_data:
                mandatory_trees = ['Events', 'RunHeaders', 'RunTails']
                trees_in_file = [tree in input_data for tree in mandatory_trees]
                if not all(trees_in_file):
                    is_magic_root_file = False
        except ValueError:
            # uproot raises ValueError if the file is not a ROOT file
            is_magic_root_file = False

        return is_magic_root_file

    @staticmethod
    def get_run_info_from_name(file_name):
        """
        This internal method extracts the run number and
        type (data/MC) from the specified file name.

        Parameters
        ----------
        file_name : str
            A file name to process.

        Returns
        -------
        run_number: int
            The run number of the file.
        is_mc: Bool
            Flag to tag MC files
        telescope: int
            Number of the telescope
        datalevel: MARSDataLevel
            Data level according to MARS

        Raises
        ------
        IndexError
            Description
        """

        mask_data_calibrated = r"\d{6}_M(\d+)_(\d+)\.\d+_Y_.*"
        mask_data_star = r"\d{6}_M(\d+)_(\d+)\.\d+_I_.*"
        mask_data_superstar = r"\d{6}_(\d+)_S_.*"
        mask_data_melibea = r"\d{6}_(\d+)_Q_.*"
        mask_mc_calibrated = r"GA_M(\d)_za\d+to\d+_\d_(\d+)_Y_.*"
        mask_mc_star = r"GA_M(\d)_za\d+to\d+_\d_(\d+)_I_.*"
        mask_mc_superstar = r"GA_za\d+to\d+_\d_S_.*"
        mask_mc_melibea = r"GA_za\d+to\d+_\d_Q_.*"
        if re.findall(mask_data_calibrated, file_name):
            parsed_info = re.findall(mask_data_calibrated, file_name)
            telescope = int(parsed_info[0][0])
            run_number = int(parsed_info[0][1])
            datalevel = MARSDataLevel.CALIBRATED
            is_mc = False
        elif re.findall(mask_data_star, file_name):
            parsed_info = re.findall(mask_data_star, file_name)
            telescope = int(parsed_info[0][0])
            run_number = int(parsed_info[0][1])
            datalevel = MARSDataLevel.STAR
            is_mc = False
        elif re.findall(mask_data_superstar, file_name):
            parsed_info = re.findall(mask_data_superstar, file_name)
            telescope = None
            run_number = int(parsed_info[0])
            datalevel = MARSDataLevel.SUPERSTAR
            is_mc = False
        elif re.findall(mask_data_melibea, file_name):
            parsed_info = re.findall(mask_data_melibea, file_name)
            telescope = None
            run_number = int(parsed_info[0])
            datalevel = MARSDataLevel.MELIBEA
            is_mc = False
        elif re.findall(mask_mc_calibrated, file_name):
            parsed_info = re.findall(mask_mc_calibrated, file_name)
            telescope = int(parsed_info[0][0])
            run_number = int(parsed_info[0][1])
            datalevel = MARSDataLevel.CALIBRATED
            is_mc = True
        elif re.findall(mask_mc_star, file_name):
            parsed_info = re.findall(mask_mc_star, file_name)
            telescope = int(parsed_info[0][0])
            run_number = int(parsed_info[0][1])
            datalevel = MARSDataLevel.STAR
            is_mc = True
        elif re.findall(mask_mc_superstar, file_name):
            parsed_info = re.findall(mask_mc_superstar, file_name)
            telescope = None
            run_number = None
            datalevel = MARSDataLevel.SUPERSTAR
            is_mc = True
        elif re.findall(mask_mc_melibea, file_name):
            parsed_info = re.findall(mask_mc_melibea, file_name)
            telescope = None
            run_number = None
            datalevel = MARSDataLevel.MELIBEA
            is_mc = True
        else:
            raise IndexError(
                'Can not identify the run number and type (data/MC) of the file'
                '{:s}'.format(file_name))

        return run_number, is_mc, telescope, datalevel

    def parse_run_info(self):
        """
        Parses run info from the TTrees in the ROOT file

        Returns
        -------
        run_number: int
            The run number of the file
        is_mc: Bool
            Flag to tag MC files
        telescope_number: int
            Number of the telescope
        datalevel: MARSDataLevel
            Data level according to MARS
        """

        runinfo_array_list = [
            'MRawRunHeader.fRunNumber',
            'MRawRunHeader.fRunType',
            'MRawRunHeader.fTelescopeNumber',
        ]

        run_info = self.file_['RunHeaders'].arrays(
            runinfo_array_list, library="np")
        run_number = int(run_info['MRawRunHeader.fRunNumber'][0])
        run_type = int(run_info['MRawRunHeader.fRunType'][0])
        telescope_number = int(run_info['MRawRunHeader.fTelescopeNumber'][0])

        # a note about run numbers:
        # mono data has run numbers starting with 1 or 2 (telescope dependent)
        # stereo data has run numbers starting with 5
        # if both telescopes are taking data with no L3,
        # also in this case run number starts with 5 (e.g. muon runs)

        # Here the data types (from MRawRunHeader.h)
        # std data = 0
        # pedestal = 1 (_P_)
        # calibration = 2 (_C_)
        # domino calibration = 3 (_L_)
        # linearity calibration = 4 (_N_)
        # point run = 7
        # monteCarlo = 256
        # none = 65535

        mc_data_type = 256

        if run_type == mc_data_type:
            is_mc = True
        else:
            is_mc = False

        events_tree = self.file_['Events']

        melibea_trees = ['MHadronness', 'MStereoParDisp', 'MEnergyEst']
        superstar_trees = ['MHillas_1', 'MHillas_2', 'MStereoPar']
        star_trees = ['MHillas']

        datalevel = MARSDataLevel.CALIBRATED
        events_keys = events_tree.keys()
        trees_in_file = [tree in events_keys for tree in melibea_trees]
        if all(trees_in_file):
            datalevel = MARSDataLevel.MELIBEA
        trees_in_file = [tree in events_keys for tree in superstar_trees]
        if all(trees_in_file):
            datalevel = MARSDataLevel.SUPERSTAR
        trees_in_file = [tree in events_keys for tree in star_trees]
        if all(trees_in_file):
            datalevel = MARSDataLevel.STAR

        return run_number, is_mc, telescope_number, datalevel

    def parse_data_info(self):
        """
        Check if data is stereo/mono and std trigger/SUMT

        Returns
        -------
        is_stereo: Bool
            True if stereo data, False if mono
        is_sumt: Bool
            True if SUMT data, False if std trigger
        """

        prescaler_mono_nosumt = [1, 1, 0, 1, 0, 0, 0, 0]
        prescaler_mono_sumt = [0, 1, 0, 1, 0, 1, 0, 0]
        prescaler_stereo = [0, 1, 0, 1, 0, 0, 0, 1]

        # L1_table_mono = "L1_4NN"
        # L1_table_stereo = "L1_3NN"

        L3_table_nosumt = "L3T_L1L1_100_SYNC"
        L3_table_sumt = "L3T_SUMSUM_100_SYNC"

        trigger_tree = self.file_["Trigger"]
        L3T_tree = self.file_["L3T"]

        # here we take the 2nd element (if possible) because sometimes
        # the first trigger report has still the old prescaler values from a previous run
        try:
            prescaler_array = trigger_tree["MTriggerPrescFact.fPrescFact"].array(library="np")
        except AssertionError:
            LOGGER.warning("No prescaler info found. Will assume standard stereo data.")
            is_stereo = True
            is_sumt = False
            return is_stereo, is_sumt

        prescaler_size = prescaler_array.size
        if prescaler_size > 1:
            prescaler = prescaler_array[1]
        else:
            prescaler = prescaler_array[0]

        if prescaler == prescaler_mono_nosumt or prescaler == prescaler_mono_sumt:
            is_stereo = False
        elif prescaler == prescaler_stereo:
            is_stereo = True
        else:
            is_stereo = True

        is_sumt = False
        if is_stereo:
            # here we take the 2nd element for the same reason as above
            # L3Table is empty for mono data i.e. taken with one telescope only
            # if both telescopes take data with no L3, L3Table is filled anyway
            L3Table_array = L3T_tree["MReportL3T.fTablename"].array(library="np")
            L3Table_size = L3Table_array.size
            if L3Table_size > 1:
                L3Table = L3Table_array[1]
            else:
                L3Table = L3Table_array[0]

            if L3Table == L3_table_sumt:
                is_sumt = True
            elif L3Table == L3_table_nosumt:
                is_sumt = False
            else:
                is_sumt = False
        else:
            if prescaler == prescaler_mono_sumt:
                is_sumt = True

        return is_stereo, is_sumt

    def prepare_subarray_info(self):
        """
        Fill SubarrayDescription container

        Returns
        -------
        subarray: ctapipe.instrument.SubarrayDescription
            Container with telescope descriptions and positions information
        """

        # MAGIC telescope positions in m wrt. to the center of CTA simulations
        # MAGIC_TEL_POSITIONS = {
        #    1: [-27.24, -146.66, 50.00] * u.m,
        #    2: [-96.44, -96.77, 51.00] * u.m
        # }

        # MAGIC telescope positions in m wrt. to the center of MAGIC simulations, from
        # CORSIKA and reflector input card
        MAGIC_TEL_POSITIONS = {
            1: [31.80, -28.10, 0.00] * u.m,
            2: [-31.80, 28.10, 0.00] * u.m
        }

        # camera info from MAGICCam.camgeom.fits.gz file
        camera_geom = load_camera_geometry()

        pulse_shape_lo_gain = np.array([0., 1., 2., 1., 0.])
        pulse_shape_hi_gain = np.array([1., 2., 3., 2., 1.])
        pulse_shape = np.vstack((pulse_shape_lo_gain, pulse_shape_hi_gain))
        camera_readout = CameraReadout(
            camera_name='MAGICCam',
            sampling_rate=u.Quantity(1.64, u.GHz),
            reference_pulse_shape=pulse_shape,
            reference_pulse_sample_width=u.Quantity(0.5, u.ns)
        )

        camera = CameraDescription('MAGICCam', camera_geom, camera_readout)

        camera.geometry.frame = CameraFrame(focal_length=OPTICS.equivalent_focal_length)

        MAGIC_TEL_DESCRIPTION = TelescopeDescription(
            name='MAGIC', tel_type='MAGIC', optics=OPTICS, camera=camera
        )

        MAGIC_TEL_DESCRIPTIONS = {1: MAGIC_TEL_DESCRIPTION, 2: MAGIC_TEL_DESCRIPTION}

        subarray = SubarrayDescription(
            name='MAGIC',
            tel_positions=MAGIC_TEL_POSITIONS,
            tel_descriptions=MAGIC_TEL_DESCRIPTIONS
        )

        if self.allowed_tels:
            subarray = self._subarray_info.select_subarray(self.allowed_tels)

        return subarray

    @staticmethod
    def decode_version_number(version_encoded):
        """
        Decodes the version number from an integer

        Parameters
        ----------
        version_encoded : int
            Version number encoded as integer

        Returns
        -------
        version_decoded: str
            Version decoded as major.minor.patch
        """

        major_version = version_encoded >> 16
        minor_version = (version_encoded % 65536) >> 8
        patch_version = (version_encoded % 65536) % 256
        version_decoded = f'{major_version}.{minor_version}.{patch_version}'

        return version_decoded

    def parse_metadata_info(self):
        """
        Parse metadata information from ROOT file

        Returns
        -------
        metadata: dict
            Dictionary containing the metadata information:
            - run number
            - real or simulated data
            - telescope number
            - subrun number
            - source RA and DEC
            - source name (real data only)
            - observation mode (real data only)
            - MARS version
            - ROOT version
        """

        metadatainfo_array_list_runheaders = [
            'MRawRunHeader.fSubRunIndex',
            'MRawRunHeader.fSourceRA',
            'MRawRunHeader.fSourceDEC',
            'MRawRunHeader.fSourceName[80]',
            'MRawRunHeader.fObservationMode[60]',
        ]

        metadatainfo_array_list_runtails = [
            'MMarsVersion_sorcerer.fMARSVersionCode',
            'MMarsVersion_sorcerer.fROOTVersionCode',
        ]

        metadata = dict()
        metadata['run_number'] = self.run_numbers
        metadata['is_simulation'] = self.is_mc
        metadata['telescope'] = self.telescope

        meta_info_runh = self.file_['RunHeaders'].arrays(
                metadatainfo_array_list_runheaders, library="np"
        )

        metadata['subrun_number'] = int(meta_info_runh['MRawRunHeader.fSubRunIndex'][0])
        metadata['source_ra'] = meta_info_runh['MRawRunHeader.fSourceRA'][0] / \
            seconds_per_hour * degrees_per_hour * u.deg
        metadata['source_dec'] = meta_info_runh['MRawRunHeader.fSourceDEC'][0] / \
            seconds_per_hour * u.deg
        if not self.is_mc:
            src_name_array = meta_info_runh['MRawRunHeader.fSourceName[80]'][0]
            metadata['source_name'] = "".join([chr(item) for item in src_name_array if item != 0])
            obs_mode_array = meta_info_runh['MRawRunHeader.fObservationMode[60]'][0]
            metadata['observation_mode'] = "".join([chr(item) for item in obs_mode_array if item != 0])

        meta_info_runt = self.file_['RunTails'].arrays(
            metadatainfo_array_list_runtails,
            library="np"
        )

        mars_version_encoded = int(meta_info_runt['MMarsVersion_sorcerer.fMARSVersionCode'][0])
        root_version_encoded = int(meta_info_runt['MMarsVersion_sorcerer.fROOTVersionCode'][0])
        metadata['mars_version_sorcerer'] = self.decode_version_number(mars_version_encoded)
        metadata['root_version_sorcerer'] = self.decode_version_number(root_version_encoded)

        return metadata

    def parse_simulation_header(self):
        """
        Parse the simulation information from the RunHeaders tree.

        Returns
        -------
        SimulationConfigContainer
            Container filled with simulation information

        Notes
        -----
        Information is extracted from the RunHeaders tree within the ROOT file.
        Within it, the MMcCorsikaRunHeader and MMcRunHeader branches are used.
        Here below the units of the members extracted, for reference:
        * fSlopeSpec: float
        * fELowLim, fEUppLim: GeV
        * fCorsikaVersion: int
        * fHeightLev[10]: centimeter
        * fAtmosphericModel: int
        * fRandomPointingConeSemiAngle: deg
        * fImpactMax: centimeter
        * fNumSimulatedShowers: int
        * fShowerThetaMax, fShowerThetaMin: deg
        * fShowerPhiMax, fShowerPhiMin: deg
        * fCWaveUpper, fCWaveLower: nanometer
        """

        # Magnetic field values at the MAGIC site (taken from CORSIKA input cards)
        # Reference system is the CORSIKA one, where x-axis points to magnetic north
        # i.e. B y-component is 0
        # MAGIC_Bdec is the magnetic declination i.e. angle between magnetic and
        # geographic north, negative if pointing westwards, positive if pointing
        # eastwards
        # MAGIC_Binc is the magnetic field inclination
        MAGIC_Bx = u.Quantity(29.5, u.uT)
        MAGIC_Bz = u.Quantity(23.0, u.uT)
        MAGIC_Btot = np.sqrt(MAGIC_Bx**2+MAGIC_Bz**2)
        MAGIC_Bdec = u.Quantity(-7.0, u.deg).to(u.rad)
        MAGIC_Binc = u.Quantity(np.arctan2(-MAGIC_Bz.value, MAGIC_Bx.value), u.rad)

        run_header_tree = self.file_['RunHeaders']
        spectral_index = run_header_tree['MMcCorsikaRunHeader.fSlopeSpec'].array(library="np")[0]
        e_low = run_header_tree['MMcCorsikaRunHeader.fELowLim'].array(library="np")[0]
        e_high = run_header_tree['MMcCorsikaRunHeader.fEUppLim'].array(library="np")[0]
        corsika_version = run_header_tree['MMcCorsikaRunHeader.fCorsikaVersion'].array(library="np")[0]
        site_height = run_header_tree['MMcCorsikaRunHeader.fHeightLev[10]'].array(library="np")[0][0]
        atm_model = run_header_tree['MMcCorsikaRunHeader.fAtmosphericModel'].array(library="np")[0]
        if self.mars_datalevel in [MARSDataLevel.CALIBRATED, MARSDataLevel.STAR]:
            view_cone = run_header_tree['MMcRunHeader.fRandomPointingConeSemiAngle'].array(library="np")[0]
            max_impact = run_header_tree['MMcRunHeader.fImpactMax'].array(library="np")[0]
            n_showers = np.sum(run_header_tree['MMcRunHeader.fNumSimulatedShowers'].array(library="np"))
            max_zd = run_header_tree['MMcRunHeader.fShowerThetaMax'].array(library="np")[0]
            min_zd = run_header_tree['MMcRunHeader.fShowerThetaMin'].array(library="np")[0]
            max_az = run_header_tree['MMcRunHeader.fShowerPhiMax'].array(library="np")[0]
            min_az = run_header_tree['MMcRunHeader.fShowerPhiMin'].array(library="np")[0]
            max_wavelength = run_header_tree['MMcRunHeader.fCWaveUpper'].array(library="np")[0]
            min_wavelength = run_header_tree['MMcRunHeader.fCWaveLower'].array(library="np")[0]
        elif self.mars_datalevel in [MARSDataLevel.SUPERSTAR, MARSDataLevel.MELIBEA]:
            view_cone = run_header_tree['MMcRunHeader_1.fRandomPointingConeSemiAngle'].array(library="np")[0]
            max_impact = run_header_tree['MMcRunHeader_1.fImpactMax'].array(library="np")[0]
            n_showers = np.sum(run_header_tree['MMcRunHeader_1.fNumSimulatedShowers'].array(library="np"))
            max_zd = run_header_tree['MMcRunHeader_1.fShowerThetaMax'].array(library="np")[0]
            min_zd = run_header_tree['MMcRunHeader_1.fShowerThetaMin'].array(library="np")[0]
            max_az = run_header_tree['MMcRunHeader_1.fShowerPhiMax'].array(library="np")[0]
            min_az = run_header_tree['MMcRunHeader_1.fShowerPhiMin'].array(library="np")[0]
            max_wavelength = run_header_tree['MMcRunHeader_1.fCWaveUpper'].array(library="np")[0]
            min_wavelength = run_header_tree['MMcRunHeader_1.fCWaveLower'].array(library="np")[0]

        return SimulationConfigContainer(
            corsika_version=corsika_version,
            energy_range_min=u.Quantity(e_low, u.GeV).to(u.TeV),
            energy_range_max=u.Quantity(e_high, u.GeV).to(u.TeV),
            prod_site_alt=u.Quantity(site_height, u.cm).to(u.m),
            spectral_index=spectral_index,
            num_showers=n_showers,
            shower_reuse=1,
            # shower_reuse not written in the magic root file, but since the
            # sim_events already include shower reuse we artificially set it
            # to 1 (actually every shower reused 5 times for std MAGIC MC)
            shower_prog_id=1,
            prod_site_B_total=MAGIC_Btot,
            prod_site_B_declination=MAGIC_Bdec,
            prod_site_B_inclination=MAGIC_Binc,
            max_alt=u.Quantity((90. - min_zd), u.deg).to(u.rad),
            min_alt=u.Quantity((90. - max_zd), u.deg).to(u.rad),
            max_az=u.Quantity(max_az, u.deg).to(u.rad),
            min_az=u.Quantity(min_az, u.deg).to(u.rad),
            max_viewcone_radius=view_cone * u.deg,
            min_viewcone_radius=0.0 * u.deg,
            max_scatter_range=u.Quantity(max_impact, u.cm).to(u.m),
            min_scatter_range=0.0 * u.m,
            atmosphere=atm_model,
            corsika_wlen_min=min_wavelength * u.nm,
            corsika_wlen_max=max_wavelength * u.nm,
        )

    def _set_active_run(self, run_number):
        """
        This internal method sets the run that will be used for data loading.

        Parameters
        ----------
        run_number: int
            The run number to use.

        Returns
        -------
        run: MarsRun
            The run to use
        """

        run = dict()
        run['number'] = run_number
        run['read_events'] = 0
        if self.mars_datalevel == MARSDataLevel.CALIBRATED:
            run['data'] = MarsCalibratedRun(self.file_, self.is_mc)

        return run

    @property
    def subarray(self):
        return self._subarray_info

    @property
    def is_simulation(self):
        return self.is_mc

    @property
    def datalevels(self):
        return (self.datalevel, )

    @property
    def obs_ids(self):
        # ToCheck: will this be compatible in the future, e.g. with merged MC files
        return [self.run_numbers]

    def _generator(self):
        """
        The default event generator. Return the stereo event
        generator instance.

        Returns
        -------

        """

        if self.mars_datalevel == MARSDataLevel.CALIBRATED:
            return self._event_generator(generate_pedestals=self.use_pedestals)

    def _event_generator(self, generate_pedestals):
        """
        Event generator. Yields ArrayEventContainer instances, filled
        with the read event data.

        Returns
        -------

        """

        counter = 0

        # Data container - is initialized once, and data is replaced after each yield
        data = ArrayEventContainer()

        # Telescopes with data:
        tel_id = self.telescope
        tels_with_data = [tel_id, ]

        # Removing the previously read data run from memory
        if self.current_run is not None:
            if 'data' in self.current_run:
                del self.current_run['data']

        # Setting the new active run
        self.current_run = self._set_active_run(self.run_numbers)

        n_pixels = self._subarray_info.tel[tel_id].camera.geometry.n_pixels

        # Set monitoring data:
        if not self.is_mc:

            monitoring_data = self.current_run['data'].monitoring_data

            monitoring_camera = MonitoringCameraContainer()
            pedestal_info = PedestalContainer()
            badpixel_info = PixelStatusContainer()

            pedestal_info.sample_time = Time(
                monitoring_data['M{:d}'.format(tel_id)]['PedestalUnix'], format='unix', scale='utc'
            )

            pedestal_info.n_events = 500  # hardcoded number of pedestal events averaged over
            pedestal_info.charge_mean = []
            pedestal_info.charge_mean.append(
                monitoring_data['M{:d}'.format(tel_id)]['PedestalFundamental']['Mean'])
            pedestal_info.charge_mean.append(
                monitoring_data['M{:d}'.format(tel_id)]['PedestalFromExtractor']['Mean'])
            pedestal_info.charge_mean.append(monitoring_data['M{:d}'.format(
                tel_id)]['PedestalFromExtractorRndm']['Mean'])
            pedestal_info.charge_std = []
            pedestal_info.charge_std.append(
                monitoring_data['M{:d}'.format(tel_id)]['PedestalFundamental']['Rms'])
            pedestal_info.charge_std.append(
                monitoring_data['M{:d}'.format(tel_id)]['PedestalFromExtractor']['Rms'])
            pedestal_info.charge_std.append(
                monitoring_data['M{:d}'.format(tel_id)]['PedestalFromExtractorRndm']['Rms'])

            t_range = Time(monitoring_data['M{:d}'.format(tel_id)]['badpixelinfoUnixRange'], format='unix', scale='utc')

            badpixel_info.hardware_failing_pixels = monitoring_data['M{:d}'.format(tel_id)]['badpixelinfo']
            badpixel_info.sample_time_range = t_range

            monitoring_camera.pedestal = pedestal_info
            monitoring_camera.pixel_status = badpixel_info

            data.mon.tel[tel_id] = monitoring_camera

        if generate_pedestals:
            if tel_id == 1:
                n_events = self.current_run['data'].n_pedestal_events_m1
            else:
                n_events = self.current_run['data'].n_pedestal_events_m2
        else:
            if tel_id == 1:
                n_events = self.current_run['data'].n_cosmics_stereo_events_m1
            else:
                n_events = self.current_run['data'].n_cosmics_stereo_events_m2

        if generate_pedestals:
            event_data = self.current_run['data'].pedestal_events[f"M{tel_id}"]
        else:
            event_data = self.current_run['data'].cosmics_stereo_events[f"M{tel_id}"]

        # Loop over the events
        for event_i in range(n_events):

            data.meta['origin'] = 'MAGIC'
            data.meta['input_url'] = self.input_url
            data.meta['max_events'] = self.max_events

            # Event and run ids
            event_id = event_data['stereo_event_number'][event_i]
            obs_id = self.current_run['number']

            data.trigger.event_type = MAGIC_TO_CTA_EVENT_TYPE.get(event_data['trigger_pattern'][event_i])
            data.trigger.tels_with_trigger = tels_with_data

            if self.allowed_tels:

                data.trigger.tels_with_trigger = np.intersect1d(
                    data.trigger.tels_with_trigger,
                    self.subarray.tel_ids,
                    assume_unique=True
                )

            if not self.is_mc:

                data.trigger.tel[tel_id] = TelescopeTriggerContainer(
                    time=Time(
                        event_data['unix'][event_i],
                        format='unix',
                        scale='utc'
                    )
                )

            # Event counter
            data.count = counter
            data.index.obs_id = obs_id
            data.index.event_id = event_id

            # Setting up the R0 container
            data.r0.tel.clear()
            data.r1.tel.clear()
            data.dl0.tel.clear()
            data.dl1.tel.clear()
            data.pointing.tel.clear()

            # Creating the telescope pointing container
            pointing = PointingContainer()
            pointing_tel = TelescopePointingContainer(
                azimuth=np.deg2rad(event_data['pointing_az'][event_i]) * u.rad,
                altitude=np.deg2rad(90 - event_data['pointing_zd'][event_i]) * u.rad,
            )

            pointing.tel[tel_id] = pointing_tel

            pointing.array_azimuth = np.deg2rad(event_data['pointing_az'][event_i])*u.rad
            pointing.array_altitude = np.deg2rad(90 - event_data['pointing_zd'][event_i])*u.rad
            pointing.array_ra = np.deg2rad(event_data['pointing_ra'][event_i])*u.rad
            pointing.array_dec = np.deg2rad(event_data['pointing_dec'][event_i])*u.rad

            data.pointing = pointing

            # Adding event charge and peak positions per pixel
            data.dl1.tel[tel_id].image = np.array(event_data['image'][event_i][:n_pixels], dtype=np.float32)
            data.dl1.tel[tel_id].peak_time = np.array(event_data['pulse_time'][event_i][:n_pixels], dtype=np.float32)

            if self.is_mc:
                # check meaning of 7deg transformation (I.Vovk)
                # adding a 7deg rotation between the orientation of corsika (x axis = magnetic north) and MARS (x axis = geographical north) frames
                # magnetic north is 7 deg westward w.r.t. geographical north
                data.simulation = SimulatedEventContainer()
                MAGIC_Bdec = self.simulation_config["prod_site_B_declination"]
                data.simulation.shower = SimulatedShowerContainer(
                    energy=u.Quantity(event_data['true_energy'][event_i], u.GeV),
                    alt=Angle((np.pi/2 - event_data['true_zd'][event_i]), u.rad),
                    az=Angle(-1 * (event_data['true_az'][event_i] - (np.pi/2 + MAGIC_Bdec.value)), u.rad),
                    shower_primary_id=(1 - event_data['true_shower_primary_id'][event_i]),
                    h_first_int=u.Quantity(event_data['true_h_first_int'][event_i], u.cm),
                    core_x=u.Quantity((event_data['true_core_x'][event_i]*np.cos(-MAGIC_Bdec) - event_data['true_core_y'][event_i]*np.sin(-MAGIC_Bdec)).value, u.cm),
                    core_y=u.Quantity((event_data['true_core_x'][event_i]*np.sin(-MAGIC_Bdec) + event_data['true_core_y'][event_i]*np.cos(-MAGIC_Bdec)).value, u.cm),
                )

            yield data
            counter += 1

        return


class MarsCalibratedRun:
    """
    This class implements reading of the event data from a single MAGIC calibrated run.
    """

    def __init__(self, uproot_file, is_mc):
        """
        Constructor of the class. Defines the run to use and the camera pixel arrangement.

        Parameters
        ----------
        uproot_file: str
            A file opened by uproot via uproot.open(file_name)
        """

        self.n_camera_pixels = 1039

        self.file_name = uproot_file.file_path

        self.is_mc = is_mc

        if '_M1_' in self.file_name:
            m1_events = self.load_events(
                uproot_file, self.is_mc, self.n_camera_pixels)
            m2_events = self.load_events(
                None, self.is_mc, self.n_camera_pixels)
        if '_M2_' in self.file_name:
            m1_events = self.load_events(
                None, self.is_mc, self.n_camera_pixels)
            m2_events = self.load_events(
                uproot_file, self.is_mc, self.n_camera_pixels)

        # Getting the event data
        self.cosmics_stereo_events = dict()
        self.cosmics_stereo_events['M1'] = m1_events["cosmics_stereo_events"]
        self.cosmics_stereo_events['M2'] = m2_events["cosmics_stereo_events"]

        self.pedestal_events = dict()
        self.pedestal_events['M1'] = m1_events["pedestal_events"]
        self.pedestal_events['M2'] = m2_events["pedestal_events"]

        # Getting the monitoring data
        self.monitoring_data = dict()
        self.monitoring_data['M1'] = m1_events["monitoring_data"]
        self.monitoring_data['M2'] = m2_events["monitoring_data"]

    @property
    def n_cosmics_stereo_events_m1(self):
        return len(self.cosmics_stereo_events['M1']['trigger_pattern'])

    @property
    def n_cosmics_stereo_events_m2(self):
        return len(self.cosmics_stereo_events['M2']['trigger_pattern'])

    @property
    def n_cosmics_mono_events_m1(self):
        return len(self.cosmics_stereo_events['M1']['trigger_pattern'])

    @property
    def n_cosmics_mono_events_m2(self):
        return len(self.cosmics_stereo_events['M2']['trigger_pattern'])

    @property
    def n_pedestal_events_m1(self):
        return len(self.pedestal_events['M1']['trigger_pattern'])

    @property
    def n_pedestal_events_m2(self):
        return len(self.pedestal_events['M2']['trigger_pattern'])

    @staticmethod
    def load_events(uproot_file, is_mc, n_camera_pixels):
        """
        This method loads events and monitoring data from the pre-defiled file
        and returns them as a dictionary.

        Parameters
        ----------
        file_name: str
            Name of the MAGIC calibrated file to use.
        is_mc: boolean
            Specify whether Monte Carlo (True) or data (False) events are read
        n_camera_pixels: int
            Number of MAGIC camera pixels (not hardcoded, but specified solely via
            ctapipe.instrument.CameraGeometry)

        Returns
        -------
        dict:
            A dictionary with the even properties: charge / arrival time data, trigger,
            direction etc.
        """

        evt_common_list = [
            'MArrivalTime.fData',
            'MCerPhotEvt.fPixels.fPhot',
            'MRawEvtHeader.fDAQEvtNumber',
            'MRawEvtHeader.fStereoEvtNumber',
            'MTriggerPattern.fPrescaled',
            'MTriggerPattern.fSkipEvent',
        ]

        # Separately, because only used with pre-processed MARS data
        # to create MPointingPos container
        pointing_array_list = [
            'MPointingPos.fZd',
            'MPointingPos.fAz',
            'MPointingPos.fRa',
            'MPointingPos.fDec',
            'MPointingPos.fDevZd',
            'MPointingPos.fDevAz',
            'MPointingPos.fDevHa',
            'MPointingPos.fDevDec',
        ]

        # Info only applicable for data:
        time_array_list = [
            'MTime.fMjd',
            'MTime.fTime.fMilliSec',
            'MTime.fNanoSec',
        ]

        drive_array_list = [
            'MReportDrive.fMjd',
            'MReportDrive.fCurrentZd',
            'MReportDrive.fCurrentAz',
            'MReportDrive.fRa',
            'MReportDrive.fDec'
        ]

        pedestal_array_list = [
            'MTimePedestals.fMjd',
            'MTimePedestals.fTime.fMilliSec',
            'MTimePedestals.fNanoSec',
            'MPedPhotFundamental.fArray.fMean',
            'MPedPhotFundamental.fArray.fRms',
            'MPedPhotFromExtractor.fArray.fMean',
            'MPedPhotFromExtractor.fArray.fRms',
            'MPedPhotFromExtractorRndm.fArray.fMean',
            'MPedPhotFromExtractorRndm.fArray.fRms'
        ]

        # Info only applicable for MC:
        mc_list = [
            'MMcEvt.fEnergy',
            'MMcEvt.fTheta',
            'MMcEvt.fPhi',
            'MMcEvt.fPartId',
            'MMcEvt.fZFirstInteraction',
            'MMcEvt.fCoreX',
            'MMcEvt.fCoreY'
        ]

        event_data = dict()

        # monitoring information (updated from time to time)
        event_data["monitoring_data"] = dict()

        event_data["monitoring_data"]['badpixelinfo'] = []
        event_data["monitoring_data"]['badpixelinfoUnixRange'] = []
        event_data["monitoring_data"]['PedestalUnix'] = np.array([])
        event_data["monitoring_data"]['PedestalFundamental'] = dict()
        event_data["monitoring_data"]['PedestalFundamental']['Mean'] = []
        event_data["monitoring_data"]['PedestalFundamental']['Rms'] = []
        event_data["monitoring_data"]['PedestalFromExtractor'] = dict()
        event_data["monitoring_data"]['PedestalFromExtractor']['Mean'] = []
        event_data["monitoring_data"]['PedestalFromExtractor']['Rms'] = []
        event_data["monitoring_data"]['PedestalFromExtractorRndm'] = dict()
        event_data["monitoring_data"]['PedestalFromExtractorRndm']['Mean'] = []
        event_data["monitoring_data"]['PedestalFromExtractorRndm']['Rms'] = []

        event_types = ["cosmics_stereo_events", "pedestal_events"]

        for event_type in event_types:

            event_data[event_type] = dict()

            event_data[event_type]['image'] = np.array([])
            event_data[event_type]['pulse_time'] = np.array([])
            event_data[event_type]['trigger_pattern'] = np.array([], dtype=np.int32)
            event_data[event_type]['stereo_event_number'] = np.array([], dtype=np.int32)
            event_data[event_type]['pointing_zd'] = np.array([])
            event_data[event_type]['pointing_az'] = np.array([])
            event_data[event_type]['pointing_ra'] = np.array([])
            event_data[event_type]['pointing_dec'] = np.array([])
            event_data[event_type]['unix'] = np.array([])

        # if no file in the list (e.g. when reading mono information), then simply
        # return empty dicts/array
        if uproot_file is None:
            return event_data

        event_data["filename"] = uproot_file.file_path

        drive_data = dict()
        drive_data['mjd'] = np.array([])
        drive_data['zd'] = np.array([])
        drive_data['az'] = np.array([])
        drive_data['ra'] = np.array([])
        drive_data['dec'] = np.array([])

        input_file = uproot_file

        pedestal_cut = f"(MTriggerPattern.fPrescaled == {PEDESTAL_TRIGGER_PATTERN})"

        if is_mc:
            cosmics_stereo_cut = f"(MTriggerPattern.fPrescaled == {MC_STEREO_TRIGGER_PATTERN}) & (MRawEvtHeader.fStereoEvtNumber != 0)"
        else:
            cosmics_stereo_cut = f"(MTriggerPattern.fPrescaled == {DATA_STEREO_TRIGGER_PATTERN})"

        events_cut = {
            "cosmics_stereo_events": cosmics_stereo_cut,
            "pedestal_events": pedestal_cut,
        }

        if not is_mc:
            # Getting the telescope drive info
            drive = input_file['Drive'].arrays(drive_array_list, library="np")

            drive_mjd = drive['MReportDrive.fMjd']
            drive_zd = drive['MReportDrive.fCurrentZd']
            drive_az = drive['MReportDrive.fCurrentAz']
            drive_ra = drive['MReportDrive.fRa'] * degrees_per_hour
            drive_dec = drive['MReportDrive.fDec']

            drive_data['mjd'] = np.concatenate((drive_data['mjd'], drive_mjd))
            drive_data['zd'] = np.concatenate((drive_data['zd'], drive_zd))
            drive_data['az'] = np.concatenate((drive_data['az'], drive_az))
            drive_data['ra'] = np.concatenate((drive_data['ra'], drive_ra))
            drive_data['dec'] = np.concatenate((drive_data['dec'], drive_dec))

            if len(drive_mjd) < 3:
                LOGGER.warning(f"File {uproot_file.file_path} has only {len(drive_mjd)} drive reports.")
                if len(drive_mjd) == 0:
                    raise MissingDriveReportError(f"File {uproot_file.file_path} does not have any drive report. Check if it was merpped correctly.")

            # get only drive reports with unique times, otherwise interpolation fails.
            drive_mjd_unique, unique_indices = np.unique(
                drive_data['mjd'],
                return_index=True
            )
            drive_zd_unique = drive_data['zd'][unique_indices]
            drive_az_unique = drive_data['az'][unique_indices]
            drive_ra_unique = drive_data['ra'][unique_indices]
            drive_dec_unique = drive_data['dec'][unique_indices]

            # Creating azimuth and zenith angles interpolators
            drive_zd_pointing_interpolator = scipy.interpolate.interp1d(
                drive_mjd_unique, drive_zd_unique, fill_value="extrapolate")
            drive_az_pointing_interpolator = scipy.interpolate.interp1d(
                drive_mjd_unique, drive_az_unique, fill_value="extrapolate")

            # Creating RA and DEC interpolators
            drive_ra_pointing_interpolator = scipy.interpolate.interp1d(
                drive_mjd_unique, drive_ra_unique, fill_value="extrapolate")
            drive_dec_pointing_interpolator = scipy.interpolate.interp1d(
                drive_mjd_unique, drive_dec_unique, fill_value="extrapolate")

        for event_type in event_types:

            if event_type == "pedestal_events" and is_mc:
                continue

            events = input_file['Events'].arrays(
                expressions=evt_common_list,
                cut=events_cut[event_type],
                library="np"
            )

            # Reading the info common to MC and real data
            charge = events['MCerPhotEvt.fPixels.fPhot']
            arrival_time = events['MArrivalTime.fData']
            trigger_pattern = events['MTriggerPattern.fPrescaled']
            stereo_event_number = events['MRawEvtHeader.fStereoEvtNumber']

            if not is_mc:

                # Reading event timing information:
                event_times = input_file['Events'].arrays(
                    expressions=time_array_list,
                    cut=events_cut[event_type],
                    library="np"
                )

                # Computing the event arrival time
                event_obs_day = Time(event_times['MTime.fMjd'], format='mjd', scale='utc')
                event_obs_day = np.round(event_obs_day.to_value(format='unix', subfmt='float'))
                event_obs_day = np.array([Decimal(str(x)) for x in event_obs_day])

                event_millisec = np.round(event_times['MTime.fTime.fMilliSec'] * msec2sec, 3)
                event_millisec = np.array([Decimal(str(x)) for x in event_millisec])

                event_nanosec = np.round(event_times['MTime.fNanoSec'] * nsec2sec, 7)
                event_nanosec = np.array([Decimal(str(x)) for x in event_nanosec])

                event_unix = event_obs_day + event_millisec + event_nanosec
                event_data[event_type]['unix'] = np.concatenate((event_data[event_type]['unix'], event_unix))
                event_mjd = Time(event_data[event_type]['unix'], format='unix', scale='utc').to_value(format='mjd', subfmt='float')

                first_drive_report_time = Time(drive_mjd_unique[0], scale='utc', format='mjd')
                last_drive_report_time = Time(drive_mjd_unique[-1], scale='utc', format='mjd')

                LOGGER.warning(f"Interpolating {event_type.replace('_', ' ')} information from {len(drive_data['mjd'])} drive reports.")
                LOGGER.warning(f"Drive reports available from {first_drive_report_time.iso} to {last_drive_report_time.iso}.")

                # Interpolating the drive pointing to the event time stamps
                event_data[event_type]['pointing_zd'] = drive_zd_pointing_interpolator(event_mjd)
                event_data[event_type]['pointing_az'] = drive_az_pointing_interpolator(event_mjd)
                event_data[event_type]['pointing_ra'] = drive_ra_pointing_interpolator(event_mjd)
                event_data[event_type]['pointing_dec'] = drive_dec_pointing_interpolator(event_mjd)

            # Reading pointing information (in units of degrees):
            else:
                # Retrieving the telescope pointing direction
                pointing = input_file['Events'].arrays(
                    expressions=pointing_array_list,
                    cut=events_cut[event_type],
                    library="np"
                )

                pointing_zd = pointing['MPointingPos.fZd'] - \
                    pointing['MPointingPos.fDevZd']
                pointing_az = pointing['MPointingPos.fAz'] - \
                    pointing['MPointingPos.fDevAz']
                # N.B. the positive sign here, as HA = local sidereal time - ra
                pointing_ra = (pointing['MPointingPos.fRa'] +
                               pointing['MPointingPos.fDevHa']) * degrees_per_hour
                pointing_dec = pointing['MPointingPos.fDec'] - \
                    pointing['MPointingPos.fDevDec']

                event_data[event_type]['pointing_zd'] = np.concatenate(
                    (event_data[event_type]['pointing_zd'], pointing_zd))
                event_data[event_type]['pointing_az'] = np.concatenate(
                    (event_data[event_type]['pointing_az'], pointing_az))
                event_data[event_type]['pointing_ra'] = np.concatenate(
                    (event_data[event_type]['pointing_ra'], pointing_ra))
                event_data[event_type]['pointing_dec'] = np.concatenate(
                    (event_data[event_type]['pointing_dec'], pointing_dec))

                mc_info = input_file['Events'].arrays(
                    expressions=mc_list,
                    cut=events_cut[event_type],
                    library="np"
                )
                # N.B.: For MC, there is only one subrun, so do not need to 'append'
                event_data[event_type]['true_energy'] = mc_info['MMcEvt.fEnergy']
                event_data[event_type]['true_zd'] = mc_info['MMcEvt.fTheta']
                event_data[event_type]['true_az'] = mc_info['MMcEvt.fPhi']
                event_data[event_type]['true_shower_primary_id'] = mc_info['MMcEvt.fPartId']
                event_data[event_type]['true_h_first_int'] = mc_info['MMcEvt.fZFirstInteraction']
                event_data[event_type]['true_core_x'] = mc_info['MMcEvt.fCoreX']
                event_data[event_type]['true_core_y'] = mc_info['MMcEvt.fCoreY']

            event_data[event_type]['image'] = np.concatenate(
                (event_data[event_type]['image'], charge))
            event_data[event_type]['pulse_time'] = np.concatenate(
                (event_data[event_type]['pulse_time'], arrival_time))
            event_data[event_type]['trigger_pattern'] = np.concatenate(
                (event_data[event_type]['trigger_pattern'], trigger_pattern))
            event_data[event_type]['stereo_event_number'] = np.concatenate(
                (event_data[event_type]['stereo_event_number'], stereo_event_number))

        if not is_mc:
            badpixelinfo = input_file['RunHeaders']['MBadPixelsCam.fArray.fInfo'].array(
                uproot.interpretation.jagged.AsJagged(
                    uproot.interpretation.numerical.AsDtype(np.dtype('>i4'))
                ), library="np")[0].reshape((4, 1183), order='F')

            # now we have 4 axes:
            # 0st axis: empty (?)
            # 1st axis: Unsuitable pixels
            # 2nd axis: Uncalibrated pixels (says why pixel is unsuitable)
            # 3rd axis: Bad hardware pixels (says why pixel is unsuitable)
            # Each axis cointains a 32bit integer encoding more information about the
            # specific problem, see MARS software, MBADPixelsPix.h
            # take first axis
            unsuitable_pix_bitinfo = badpixelinfo[1][:n_camera_pixels]
            # extract unsuitable bit:
            unsuitable_pix = np.zeros(n_camera_pixels, dtype=np.bool)
            for i in range(n_camera_pixels):
                unsuitable_pix[i] = int('\t{0:08b}'.format(
                    unsuitable_pix_bitinfo[i] & 0xff)[-2])
            event_data["monitoring_data"]['badpixelinfo'].append(unsuitable_pix)
            # save time interval of badpixel info:
            event_data["monitoring_data"]['badpixelinfoUnixRange'].append([event_unix[0], event_unix[-1]])

        # try to read Pedestals tree (soft fail if not present)
            try:
                pedestal_info = input_file['Pedestals'].arrays(
                    expressions=pedestal_array_list,
                    library="np"
                )

                pedestal_obs_day = Time(pedestal_info['MTimePedestals.fMjd'], format='mjd', scale='utc')
                pedestal_obs_day = np.round(pedestal_obs_day.to_value(format='unix', subfmt='float'))
                pedestal_obs_day = np.array([Decimal(str(x)) for x in pedestal_obs_day])

                pedestal_millisec = np.round(pedestal_info['MTimePedestals.fTime.fMilliSec'] * msec2sec, 3)
                pedestal_millisec = np.array([Decimal(str(x)) for x in pedestal_millisec])

                pedestal_nanosec = np.round(pedestal_info['MTimePedestals.fNanoSec'] * nsec2sec, 7)
                pedestal_nanosec = np.array([Decimal(str(x)) for x in pedestal_nanosec])

                pedestal_unix = pedestal_obs_day + pedestal_millisec + pedestal_nanosec
                event_data["monitoring_data"]['PedestalUnix'] = np.concatenate((event_data["monitoring_data"]['PedestalUnix'], pedestal_unix))

                n_pedestals = len(pedestal_unix)

                for quantity in ['Mean', 'Rms']:
                    for i_pedestal in range(n_pedestals):
                        event_data["monitoring_data"]['PedestalFundamental'][quantity].append(
                            pedestal_info[f'MPedPhotFundamental.fArray.f{quantity}'][i_pedestal][:n_camera_pixels])
                        event_data["monitoring_data"]['PedestalFromExtractor'][quantity].append(
                            pedestal_info[f'MPedPhotFromExtractor.fArray.f{quantity}'][i_pedestal][:n_camera_pixels])
                        event_data["monitoring_data"]['PedestalFromExtractorRndm'][quantity].append(
                            pedestal_info[f'MPedPhotFromExtractorRndm.fArray.f{quantity}'][i_pedestal][:n_camera_pixels])

            except KeyError:
                LOGGER.warning(
                    "Pedestals tree not present in file. Cleaning algorithm may fail.")

            event_data["monitoring_data"]['badpixelinfo'] = np.array(event_data["monitoring_data"]['badpixelinfo'])
            event_data["monitoring_data"]['badpixelinfoUnixRange'] = np.array(event_data["monitoring_data"]['badpixelinfoUnixRange'])
            # sort monitoring data:
            order = np.argsort(event_data["monitoring_data"]['PedestalUnix'])
            event_data["monitoring_data"]['PedestalUnix'] = event_data["monitoring_data"]['PedestalUnix'][order]

            for quantity in ['Mean', 'Rms']:
                event_data["monitoring_data"]['PedestalFundamental'][quantity] = np.array(
                    event_data["monitoring_data"]['PedestalFundamental'][quantity])
                event_data["monitoring_data"]['PedestalFromExtractor'][quantity] = np.array(
                    event_data["monitoring_data"]['PedestalFromExtractor'][quantity])
                event_data["monitoring_data"]['PedestalFromExtractorRndm'][quantity] = np.array(
                    event_data["monitoring_data"]['PedestalFromExtractorRndm'][quantity])

        if not is_mc:
            stereo_event_number = event_data["cosmics_stereo_events"]["stereo_event_number"]

            max_total_jumps = 100

            # check for bit flips in the stereo event ID:
            event_difference = np.diff(stereo_event_number.astype(int))
            event_difference_id = np.where(event_difference < 0)[0]

            if len(event_difference_id) > 0:
                LOGGER.warning(f'Warning: detected {len(event_difference_id)} bitflips in file {event_data["filename"]}')
                total_jumped_events = 0
                for i in event_difference_id:
                    jumped_events = int(stereo_event_number[i])-int(stereo_event_number[i+1])
                    total_jumped_events += jumped_events
                    LOGGER.warning(
                        f"Jump of L3 number backward from {stereo_event_number[i]} to "
                        f"{stereo_event_number[i+1]}; total jumped events so far: "
                        f"{total_jumped_events}"
                    )

                if total_jumped_events > max_total_jumps:
                    LOGGER.warning(
                        f"More than {max_total_jumps} in stereo trigger number; "
                        f"you may have to match events by timestamp at a later stage."
                    )

        return event_data


class PixelStatusContainer(Container):
    """
    Container for pixel status information
    It contains masks obtained by several data analysis steps
    At r0/r1 level only the hardware_mask is initialized
    """

    sample_time_range = Field(
        [], "Range of time of the pedestal events [t_min, t_max]", unit=u.s
    )

    hardware_failing_pixels = Field(
        None,
        "Boolean np array (True = failing pixel) from the hardware pixel status data ("
        "n_chan, n_pix)",
    )

    pedestal_failing_pixels = Field(
        None,
        "Boolean np array (True = failing pixel) from the pedestal data analysis ("
        "n_chan, n_pix)",
    )

    flatfield_failing_pixels = Field(
        None,
        "Boolean np array (True = failing pixel) from the flat-field data analysis ("
        "n_chan, n_pix)",
    )
