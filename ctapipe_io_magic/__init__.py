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
from decimal import Decimal
from enum import Enum, auto
from astropy.coordinates import Angle
from astropy import units as u
from astropy.time import Time

from ctapipe.io.eventsource import EventSource
from ctapipe.io.datalevels import DataLevel
from ctapipe.core import Container, Field
from ctapipe.core.traits import Bool
from ctapipe.coordinates import CameraFrame

from ctapipe.containers import (
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
    CameraReadout,
)

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

# MAGIC telescope description
OPTICS = OpticsDescription.from_name('MAGIC')
MAGICCAM = CameraDescription.from_name("MAGICCam")
pulse_shape_lo_gain = np.array([0., 1., 2., 1., 0.])
pulse_shape_hi_gain = np.array([1., 2., 3., 2., 1.])
pulse_shape = np.vstack((pulse_shape_lo_gain, pulse_shape_lo_gain))
MAGICCAM.readout = CameraReadout(
    camera_name='MAGICCam',
    sampling_rate=u.Quantity(1.64, u.GHz),
    reference_pulse_shape=pulse_shape,
    reference_pulse_sample_width=u.Quantity(0.5, u.ns)
)
MAGICCAM.geometry.frame = CameraFrame(focal_length=OPTICS.equivalent_focal_length)
GEOM = MAGICCAM.geometry
MAGIC_TEL_DESCRIPTION = TelescopeDescription(
    name='MAGIC', tel_type='MAGIC', optics=OPTICS, camera=MAGICCAM)
MAGIC_TEL_DESCRIPTIONS = {1: MAGIC_TEL_DESCRIPTION, 2: MAGIC_TEL_DESCRIPTION}


class MARSDataLevel(Enum):
    """
    Enum of the different MARS Data Levels
    """

    CALIBRATED = auto()  # Calibrated images in charge and time (no waveforms)
    STAR = auto()  # Cleaned images, with Hillas parametrization
    SUPERSTAR = auto()  # Stereo parameters reconstructed
    MELIBEA = auto()  # Reconstruction of hadronness, event direction and energy


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

        self._subarray_info = SubarrayDescription(
            name='MAGIC',
            tel_positions=MAGIC_TEL_POSITIONS,
            tel_descriptions=MAGIC_TEL_DESCRIPTIONS
        )
        if self.allowed_tels:
            self._subarray_info = self._subarray_info.select_subarray(self.allowed_tels)

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
        mask_data_star       = r"\d{6}_M(\d+)_(\d+)\.\d+_I_.*"
        mask_data_superstar  = r"\d{6}_(\d+)_S_.*"
        mask_data_melibea    = r"\d{6}_(\d+)_Q_.*"
        mask_mc_calibrated   = r"GA_M(\d)_za\d+to\d+_\d_(\d+)_Y_.*"
        mask_mc_star         = r"GA_M(\d)_za\d+to\d+_\d_(\d+)_I_.*"
        mask_mc_superstar    = r"GA_za\d+to\d+_\d_S_.*"
        mask_mc_melibea      = r"GA_za\d+to\d+_\d_Q_.*"
        if re.findall(mask_data_calibrated, file_name):
            parsed_info = re.findall(mask_data_calibrated, file_name)
            telescope  = int(parsed_info[0][0])
            run_number = int(parsed_info[0][1])
            datalevel  = MARSDataLevel.CALIBRATED
            is_mc = False
        elif re.findall(mask_data_star, file_name):
            parsed_info = re.findall(mask_data_star, file_name)
            telescope  = int(parsed_info[0][0])
            run_number = int(parsed_info[0][1])
            datalevel  = MARSDataLevel.STAR
            is_mc = False
        elif re.findall(mask_data_superstar, file_name):
            parsed_info = re.findall(mask_data_superstar, file_name)
            telescope  = None
            run_number = int(parsed_info[0])
            datalevel  = MARSDataLevel.SUPERSTAR
            is_mc = False
        elif re.findall(mask_data_melibea, file_name):
            parsed_info = re.findall(mask_data_melibea, file_name)
            telescope  = None
            run_number = int(parsed_info[0])
            datalevel  = MARSDataLevel.MELIBEA
            is_mc = False
        elif re.findall(mask_mc_calibrated, file_name):
            parsed_info = re.findall(mask_mc_calibrated, file_name)
            telescope  = int(parsed_info[0][0])
            run_number = int(parsed_info[0][1])
            datalevel  = MARSDataLevel.CALIBRATED
            is_mc = True
        elif re.findall(mask_mc_star, file_name):
            parsed_info = re.findall(mask_mc_star, file_name)
            telescope  = int(parsed_info[0][0])
            run_number = int(parsed_info[0][1])
            datalevel  = MARSDataLevel.STAR
            is_mc = True
        elif re.findall(mask_mc_superstar, file_name):
            parsed_info = re.findall(mask_mc_superstar, file_name)
            telescope  = None
            run_number = None
            datalevel  = MARSDataLevel.SUPERSTAR
            is_mc = True
        elif re.findall(mask_mc_melibea, file_name):
            parsed_info = re.findall(mask_mc_melibea, file_name)
            telescope  = None
            run_number = None
            datalevel  = MARSDataLevel.MELIBEA
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
                metadatainfo_array_list_runtails, library="np"
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

        run_header_tree = self.file_['RunHeaders']
        spectral_index  = run_header_tree['MMcCorsikaRunHeader.fSlopeSpec'].array(library="np")[0]
        e_low           = run_header_tree['MMcCorsikaRunHeader.fELowLim'].array(library="np")[0]
        e_high          = run_header_tree['MMcCorsikaRunHeader.fEUppLim'].array(library="np")[0]
        corsika_version = run_header_tree['MMcCorsikaRunHeader.fCorsikaVersion'].array(library="np")[0]
        site_height     = run_header_tree['MMcCorsikaRunHeader.fHeightLev[10]'].array(library="np")[0][0]
        atm_model       = run_header_tree['MMcCorsikaRunHeader.fAtmosphericModel'].array(library="np")[0]
        if self.mars_datalevel in [MARSDataLevel.CALIBRATED, MARSDataLevel.STAR]:
            view_cone       = run_header_tree['MMcRunHeader.fRandomPointingConeSemiAngle'].array(library="np")[0]
            max_impact      = run_header_tree['MMcRunHeader.fImpactMax'].array(library="np")[0]
            n_showers       = np.sum(run_header_tree['MMcRunHeader.fNumSimulatedShowers'].array(library="np"))
            max_zd          = run_header_tree['MMcRunHeader.fShowerThetaMax'].array(library="np")[0]
            min_zd          = run_header_tree['MMcRunHeader.fShowerThetaMin'].array(library="np")[0]
            max_az          = run_header_tree['MMcRunHeader.fShowerPhiMax'].array(library="np")[0]
            min_az          = run_header_tree['MMcRunHeader.fShowerPhiMin'].array(library="np")[0]
            max_wavelength  = run_header_tree['MMcRunHeader.fCWaveUpper'].array(library="np")[0]
            min_wavelength  = run_header_tree['MMcRunHeader.fCWaveLower'].array(library="np")[0]
        elif self.mars_datalevel in [MARSDataLevel.SUPERSTAR, MARSDataLevel.MELIBEA]:
            view_cone       = run_header_tree['MMcRunHeader_1.fRandomPointingConeSemiAngle'].array(library="np")[0]
            max_impact      = run_header_tree['MMcRunHeader_1.fImpactMax'].array(library="np")[0]
            n_showers       = np.sum(run_header_tree['MMcRunHeader_1.fNumSimulatedShowers'].array(library="np"))
            max_zd          = run_header_tree['MMcRunHeader_1.fShowerThetaMax'].array(library="np")[0]
            min_zd          = run_header_tree['MMcRunHeader_1.fShowerThetaMin'].array(library="np")[0]
            max_az          = run_header_tree['MMcRunHeader_1.fShowerPhiMax'].array(library="np")[0]
            min_az          = run_header_tree['MMcRunHeader_1.fShowerPhiMin'].array(library="np")[0]
            max_wavelength  = run_header_tree['MMcRunHeader_1.fCWaveUpper'].array(library="np")[0]
            min_wavelength  = run_header_tree['MMcRunHeader_1.fCWaveLower'].array(library="np")[0]

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
            if self.use_pedestals:
                return self._pedestal_event_generator(telescope=f"M{self.telescope}")
            else:
                return self._mono_event_generator(telescope=f"M{self.telescope}")

    def _stereo_event_generator(self):
        """
        Stereo event generator. Yields DataContainer instances, filled
        with the read event data.

        Returns
        -------

        """

        counter = 0

        # Data container - is initialized once, and data is replaced within it after each yield
        data = ArrayEventContainer()

        # Telescopes with data:
        tels_in_file = ["m1", "m2"]
        tels_with_data = [1, 2]

        # Loop over the available data runs
        for run_number in self.run_numbers:

            # Removing the previously read data run from memory
            if self.current_run is not None:
                if 'data' in self.current_run:
                    del self.current_run['data']

            # Setting the new active run (class MarsRun object)
            self.current_run = self._set_active_run(run_number)
            
            # Set monitoring data:
            if not self.is_mc:

                monitoring_data = self.current_run['data'].monitoring_data

                for tel_i, tel_id in enumerate(tels_in_file):
                    monitoring_camera = MonitoringCameraContainer()
                    pedestal_info = PedestalContainer()
                    badpixel_info = PixelStatusContainer()

                    pedestal_info.sample_time = Time(
                        monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalUnix'], format='unix', scale='utc'
                    )

                    # hardcoded number of pedestal events averaged over:
                    pedestal_info.n_events = 500
                    pedestal_info.charge_mean = []
                    pedestal_info.charge_mean.append(
                        monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalFundamental']['Mean'])
                    pedestal_info.charge_mean.append(
                        monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalFromExtractor']['Mean'])
                    pedestal_info.charge_mean.append(monitoring_data['M{:d}'.format(
                        tel_i + 1)]['PedestalFromExtractorRndm']['Mean'])
                    pedestal_info.charge_std = []
                    pedestal_info.charge_std.append(
                        monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalFundamental']['Rms'])
                    pedestal_info.charge_std.append(
                        monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalFromExtractor']['Rms'])
                    pedestal_info.charge_std.append(
                        monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalFromExtractorRndm']['Rms'])

                    t_range = Time(monitoring_data['M{:d}'.format(tel_i + 1)]['badpixelinfoUnixRange'], format='unix', scale='utc')

                    badpixel_info.hardware_failing_pixels = monitoring_data['M{:d}'.format(tel_i + 1)]['badpixelinfo']
                    badpixel_info.sample_time_range = t_range

                    monitoring_camera.pedestal = pedestal_info
                    monitoring_camera.pixel_status = badpixel_info

                    data.mon.tels_with_data = [1, 2]
                    data.mon.tel[tel_i + 1] = monitoring_camera

            else:
                assert self.current_run['data'].mcheader_data['M1'] == self.current_run['data'].mcheader_data['M2'], "Simulation configurations are different for M1 and M2 !!!"
                data.mcheader.num_showers = self.current_run['data'].mcheader_data['M1']['sim_nevents']
                data.mcheader.shower_reuse = self.current_run['data'].mcheader_data['M1']['sim_reuse']
                data.mcheader.energy_range_min = (self.current_run['data'].mcheader_data['M1']['sim_emin']).to(u.TeV) # GeV->TeV
                data.mcheader.energy_range_max = (self.current_run['data'].mcheader_data['M1']['sim_emax']).to(u.TeV) # GeV->TeV
                data.mcheader.spectral_index = self.current_run['data'].mcheader_data['M1']['sim_eslope']
                data.mcheader.max_scatter_range = (self.current_run['data'].mcheader_data['M1']['sim_max_impact']).to(u.m) # cm->m
                data.mcheader.max_viewcone_radius = (self.current_run['data'].mcheader_data['M1']['sim_conesemiangle']).to(u.deg)# deg->deg
                if data.mcheader.max_viewcone_radius != 0.:
                    data.mcheader.diffuse = True
                else:
                    data.mcheader.diffuse = False
                    

            # Loop over the events
            for event_i in range(self.current_run['data'].n_stereo_events):
                # Event and run ids
                event_order_number = self.current_run['data'].stereo_ids[event_i][0]
                event_id = self.current_run['data'].event_data['M1']['stereo_event_number'][event_order_number]
                obs_id = self.current_run['number']

                # Reading event data
                event_data = self.current_run['data'].get_stereo_event_data(event_i)

                data.meta['origin'] = 'MAGIC'
                data.meta['input_url'] = self.input_url
                data.meta['max_events'] = self.max_events

                # Event counter
                data.count = counter
                data.index.obs_id = obs_id
                data.index.event_id = event_id

                # Setting up the R0 container
                data.r0.tel.clear()

                # Setting up the R1 container
                data.r1.tel.clear()

                # Setting up the DL0 container
                data.dl0.tel.clear()

                # Setting up the DL1 container
                data.dl1.tel.clear()

                pointing = PointingContainer()
                # Filling the DL1 container with the event data
                for tel_i, tel_id in enumerate(tels_in_file):
                    # Creating the telescope pointing container
                    pointing_tel = TelescopePointingContainer()
                    
                    pointing_tel.azimuth = np.deg2rad(
                        event_data['{:s}_pointing_az'.format(tel_id)]) * u.rad
                    
                    pointing_tel.altitude = np.deg2rad(
                        90 - event_data['{:s}_pointing_zd'.format(tel_id)]) * u.rad
                    
                    # pointing.ra = np.deg2rad(
                    #    event_data['{:s}_pointing_ra'.format(tel_id)]) * u.rad
                    # pointing.dec = np.deg2rad(
                    #    event_data['{:s}_pointing_dec'.format(tel_id)]) * u.rad

                    pointing.tel[tel_i + 1] = pointing_tel

                    # Adding trigger id (MAGIC nomenclature)
                    data.r0.tel[tel_i + 1].trigger_type = self.current_run['data'].event_data['M1']['trigger_pattern'][event_order_number]
                    data.r1.tel[tel_i + 1].trigger_type = self.current_run['data'].event_data['M1']['trigger_pattern'][event_order_number]
                    data.dl0.tel[tel_i + 1].trigger_type = self.current_run['data'].event_data['M1']['trigger_pattern'][event_order_number]

                    # Adding event charge and peak positions per pixel
                    data.dl1.tel[tel_i +
                                 1].image = event_data['{:s}_image'.format(tel_id)]
                    data.dl1.tel[tel_i +
                                 1].peak_time = event_data['{:s}_pulse_time'.format(tel_id)]
                
                pointing.array_azimuth = np.deg2rad(event_data['m1_pointing_az']) * u.rad
                pointing.array_altitude = np.deg2rad(90 - event_data['m1_pointing_zd']) * u.rad
                pointing.array_ra = np.deg2rad(event_data['m1_pointing_ra']) * u.rad
                pointing.array_dec = np.deg2rad(event_data['m1_pointing_dec']) * u.rad
                data.pointing = pointing

                if not self.is_mc:

                    for tel_i, tel_id in enumerate(tels_in_file):

                        data.trigger.tel[tel_i + 1] = TelescopeTriggerContainer(
                            time=Time(event_data[f'{tel_id}_unix'], format='unix', scale='utc')
                        )

                else:
                    data.mc.energy = event_data['true_energy'] * u.GeV
                    data.mc.alt = (np.pi/2 - event_data['true_zd']) * u.rad
                    # check meaning of 7deg transformation (I.Vovk)
                    data.mc.az = -1 * \
                        (event_data['true_az'] - np.deg2rad(180 - 7)) * u.rad
                    data.mc.shower_primary_id = 1 - \
                        event_data['true_shower_primary_id']
                    data.mc.h_first_int = event_data['true_h_first_int'] * u.cm
                    
                    # adding a 7deg rotation between the orientation of corsika (x axis = magnetic north) and MARS (x axis = geographical north) frames
                    # magnetic north is 7 deg westward w.r.t. geographical north
                    rot_corsika = 7 *u.deg
                    data.mc.core_x = (event_data['true_core_x']*np.cos(rot_corsika) - event_data['true_core_y']*np.sin(rot_corsika))* u.cm
                    data.mc.core_y = (event_data['true_core_x']*np.sin(rot_corsika) + event_data['true_core_y']*np.cos(rot_corsika))* u.cm

                # Setting the telescopes with data
                data.r0.tels_with_data = tels_with_data
                data.r1.tels_with_data = tels_with_data
                data.dl0.tels_with_data = tels_with_data
                data.trigger.tels_with_trigger = tels_with_data

                yield data
                counter += 1

        return

    def _mono_event_generator(self, telescope):
        """
        Mono event generator. Yields DataContainer instances, filled
        with the read event data.

        Parameters
        ----------
        telescope: str
            The telescope for which to return events. Can be either "M1" or "M2".

        Returns
        -------

        """

        counter = 0
        telescope = telescope.upper()

        # Data container - is initialized once, and data is replaced after each yield
        data = ArrayEventContainer()

        # Telescopes with data:
        tels_in_file = ["M1", "M2"]

        if telescope not in tels_in_file:
            raise ValueError(f"Specified telescope {telescope} is not in the allowed list {tels_in_file}")

        tel_i = tels_in_file.index(telescope)
        tels_with_data = [tel_i + 1, ]

        # Removing the previously read data run from memory
        if self.current_run is not None:
            if 'data' in self.current_run:
                del self.current_run['data']

        # Setting the new active run
        self.current_run = self._set_active_run(self.run_numbers)

        # Set monitoring data:
        if not self.is_mc:

            monitoring_data = self.current_run['data'].monitoring_data

            monitoring_camera = MonitoringCameraContainer()
            pedestal_info = PedestalContainer()
            badpixel_info = PixelStatusContainer()

            pedestal_info.sample_time = Time(
                monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalUnix'], format='unix', scale='utc'
            )

            pedestal_info.n_events = 500  # hardcoded number of pedestal events averaged over
            pedestal_info.charge_mean = []
            pedestal_info.charge_mean.append(
                monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalFundamental']['Mean'])
            pedestal_info.charge_mean.append(
                monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalFromExtractor']['Mean'])
            pedestal_info.charge_mean.append(monitoring_data['M{:d}'.format(
                tel_i + 1)]['PedestalFromExtractorRndm']['Mean'])
            pedestal_info.charge_std = []
            pedestal_info.charge_std.append(
                monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalFundamental']['Rms'])
            pedestal_info.charge_std.append(
                monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalFromExtractor']['Rms'])
            pedestal_info.charge_std.append(
                monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalFromExtractorRndm']['Rms'])

            t_range = Time(monitoring_data['M{:d}'.format(tel_i + 1)]['badpixelinfoUnixRange'], format='unix', scale='utc')

            badpixel_info.hardware_failing_pixels = monitoring_data['M{:d}'.format(tel_i + 1)]['badpixelinfo']
            badpixel_info.sample_time_range = t_range

            monitoring_camera.pedestal = pedestal_info
            monitoring_camera.pixel_status = badpixel_info

            data.mon.tel[tel_i + 1] = monitoring_camera

        if telescope == 'M1':
            n_events = self.current_run['data'].n_mono_events_m1
        else:
            n_events = self.current_run['data'].n_mono_events_m2

        # Loop over the events
        for event_i in range(n_events):

            # Event and run ids
            event_order_number = self.current_run['data'].mono_ids[telescope][event_i]
            event_id = self.current_run['data'].event_data[telescope]['stereo_event_number'][event_order_number]
            obs_id = self.current_run['number']

            # Reading event data
            event_data = self.current_run['data'].get_mono_event_data(event_i, telescope=telescope)

            data.meta['origin'] = 'MAGIC'
            data.meta['input_url'] = self.input_url
            data.meta['max_events'] = self.max_events

            data.trigger.event_type = self.current_run['data'].event_data[telescope]['trigger_pattern'][event_order_number]
            data.trigger.tels_with_trigger = tels_with_data

            if self.allowed_tels:

                data.trigger.tels_with_trigger = np.intersect1d(
                    data.trigger.tels_with_trigger,
                    self.subarray.tel_ids,
                    assume_unique=True
                )
            
            if not self.is_mc:

                data.trigger.tel[tel_i + 1] = TelescopeTriggerContainer(
                    time=Time(event_data['unix'], format='unix', scale='utc')
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
                azimuth=np.deg2rad(event_data['pointing_az']) * u.rad,
                altitude=np.deg2rad(90 - event_data['pointing_zd']) * u.rad,)

            pointing.tel[tel_i + 1] = pointing_tel

            pointing.array_azimuth = np.deg2rad(event_data['pointing_az']) * u.rad
            pointing.array_altitude = np.deg2rad(90 - event_data['pointing_zd']) * u.rad
            pointing.array_ra = np.deg2rad(event_data['pointing_ra']) * u.rad
            pointing.array_dec = np.deg2rad(event_data['pointing_dec']) * u.rad

            data.pointing = pointing

            # Adding event charge and peak positions per pixel
            data.dl1.tel[tel_i + 1].image = event_data['image']
            data.dl1.tel[tel_i + 1].peak_time = event_data['pulse_time']

            if self.is_mc:
                # check meaning of 7deg transformation (I.Vovk)
                # adding a 7deg rotation between the orientation of corsika (x axis = magnetic north) and MARS (x axis = geographical north) frames
                # magnetic north is 7 deg westward w.r.t. geographical north
                data.simulation = SimulatedEventContainer()
                data.simulation.shower = SimulatedShowerContainer(
                    energy=u.Quantity(event_data['true_energy'], u.GeV),
                    alt=Angle((np.pi/2 - event_data['true_zd']), u.rad),
                    az=Angle(-1 * (event_data['true_az'] - (np.pi/2 + MAGIC_Bdec.value)), u.rad),
                    shower_primary_id=(1 - event_data['true_shower_primary_id']),
                    h_first_int=u.Quantity(event_data['true_h_first_int'], u.cm),
                    core_x=u.Quantity((event_data['true_core_x']*np.cos(-MAGIC_Bdec) - event_data['true_core_y']*np.sin(-MAGIC_Bdec)).value, u.cm),
                    core_y=u.Quantity((event_data['true_core_x']*np.sin(-MAGIC_Bdec) + event_data['true_core_y']*np.cos(-MAGIC_Bdec)).value, u.cm),
                )

            yield data
            counter += 1

        return

    def _pedestal_event_generator(self, telescope):
        """
        Pedestal event generator. Yields DataContainer instances, filled
        with the read event data.

        Parameters
        ----------
        telescope: str
            The telescope for which to return events. Can be either "M1" or "M2".

        Returns
        -------

        """

        counter = 0
        telescope = telescope.upper()

        # Data container - is initialized once, and data is replaced after each yield
        data = ArrayEventContainer()

        # Telescopes with data:
        tels_in_file = ["M1", "M2"]

        if telescope not in tels_in_file:
            raise ValueError(f"Specified telescope {telescope} is not in the allowed list {tels_in_file}")

        tel_i = tels_in_file.index(telescope)
        tels_with_data = [tel_i + 1, ]

        # Removing the previously read data run from memory
        if self.current_run is not None:
            if 'data' in self.current_run:
                del self.current_run['data']

        # Setting the new active run
        self.current_run = self._set_active_run(self.run_numbers)

        monitoring_data = self.current_run['data'].monitoring_data

        monitoring_camera = MonitoringCameraContainer()
        pedestal_info = PedestalContainer()
        badpixel_info = PixelStatusContainer()

        pedestal_info.sample_time = Time(
            monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalUnix'], format='unix', scale='utc'
        )

        pedestal_info.n_events = 500  # hardcoded number of pedestal events averaged over
        pedestal_info.charge_mean = []
        pedestal_info.charge_mean.append(
            monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalFundamental']['Mean'])
        pedestal_info.charge_mean.append(
            monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalFromExtractor']['Mean'])
        pedestal_info.charge_mean.append(monitoring_data['M{:d}'.format(
            tel_i + 1)]['PedestalFromExtractorRndm']['Mean'])
        pedestal_info.charge_std = []
        pedestal_info.charge_std.append(
            monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalFundamental']['Rms'])
        pedestal_info.charge_std.append(
            monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalFromExtractor']['Rms'])
        pedestal_info.charge_std.append(
            monitoring_data['M{:d}'.format(tel_i + 1)]['PedestalFromExtractorRndm']['Rms'])

        t_range = Time(monitoring_data['M{:d}'.format(tel_i + 1)]['badpixelinfoUnixRange'], format='unix', scale='utc')

        badpixel_info.hardware_failing_pixels = monitoring_data['M{:d}'.format(
            tel_i + 1)]['badpixelinfo']
        badpixel_info.sample_time_range = t_range

        monitoring_camera.pedestal = pedestal_info
        monitoring_camera.pixel_status = badpixel_info

        data.mon.tel[tel_i + 1] = monitoring_camera

        if telescope == 'M1':
            n_events = self.current_run['data'].n_pedestal_events_m1
        else:
            n_events = self.current_run['data'].n_pedestal_events_m2

        # Loop over the events
        for event_i in range(n_events):
            # Event and run ids
            event_order_number = self.current_run['data'].pedestal_ids[telescope][event_i]
            event_id = self.current_run['data'].event_data[telescope]['stereo_event_number'][event_order_number]
            obs_id = self.current_run['number']

            # Reading event data
            event_data = self.current_run['data'].get_pedestal_event_data(
                event_i, telescope=telescope)

            data.meta['origin'] = 'MAGIC'
            data.meta['input_url'] = self.input_url
            data.meta['max_events'] = self.max_events

            data.trigger.event_type = self.current_run['data'].event_data[telescope]['trigger_pattern'][event_order_number]
            data.trigger.tels_with_trigger = tels_with_data

            if self.allowed_tels:
                data.trigger.tels_with_trigger = np.intersect1d(
                    data.trigger.tels_with_trigger,
                    self.subarray.tel_ids,
                    assume_unique=True,)

            if not self.is_mc:
                
                # Adding the event arrival time
                data.trigger.tel[tel_i + 1] = TelescopeTriggerContainer(
                    time=Time(event_data['unix'], format='unix', scale='utc')
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
                azimuth=np.deg2rad(event_data['pointing_az']) * u.rad,
                altitude=np.deg2rad(90 - event_data['pointing_zd']) * u.rad,)

            pointing.tel[tel_i + 1] = pointing_tel

            pointing.array_azimuth = np.deg2rad(event_data['pointing_az']) * u.rad
            pointing.array_altitude = np.deg2rad(90 - event_data['pointing_zd']) * u.rad
            pointing.array_ra = np.deg2rad(event_data['pointing_ra']) * u.rad
            pointing.array_dec = np.deg2rad(event_data['pointing_dec']) * u.rad

            data.pointing = pointing

            # Adding event charge and peak positions per pixel
            data.dl1.tel[tel_i + 1].image = event_data['image']
            data.dl1.tel[tel_i + 1].peak_time = event_data['pulse_time']

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

        self.n_camera_pixels = GEOM.n_pixels

        self.file_name = uproot_file.file_path

        self.is_mc = is_mc

        if '_M1_' in self.file_name:
            m1_data = self.load_events(
                uproot_file, self.is_mc, self.n_camera_pixels)
            m2_data = self.load_events(
                None, self.is_mc, self.n_camera_pixels)
        if '_M2_' in self.file_name:
            m1_data = self.load_events(
                None, self.is_mc, self.n_camera_pixels)
            m2_data = self.load_events(
                uproot_file, self.is_mc, self.n_camera_pixels)

        # Getting the event data
        self.event_data = dict()
        self.event_data['M1'] = m1_data[0]
        self.event_data['M2'] = m2_data[0]

        # Getting the monitoring data
        self.monitoring_data = dict()
        self.monitoring_data['M1'] = m1_data[1]
        self.monitoring_data['M2'] = m2_data[1]

        # Detecting pedestal events
        self.pedestal_ids = self._find_pedestal_events()
        # Detecting stereo events
        self.stereo_ids = self._find_stereo_events()
        # Detecting mono events
        if self.is_mc:
            self.mono_ids = self._find_stereo_mc_events()
        else:
            self.mono_ids = self._find_mono_events()

    @property
    def n_events_m1(self):
        return len(self.event_data['M1']['unix'])

    @property
    def n_events_m2(self):
        return len(self.event_data['M2']['unix'])

    @property
    def n_stereo_events(self):
        return len(self.stereo_ids)

    @property
    def n_mono_events_m1(self):
        return len(self.mono_ids['M1'])

    @property
    def n_mono_events_m2(self):
        return len(self.mono_ids['M2'])

    @property
    def n_pedestal_events_m1(self):
        return len(self.pedestal_ids['M1'])

    @property
    def n_pedestal_events_m2(self):
        return len(self.pedestal_ids['M2'])

    @staticmethod
    def load_events(uproot_file, is_mc, n_camera_pixels):
        """
        This method loads events and monitoring data from the pre-defiled file and returns them as a dictionary.

        Parameters
        ----------
        file_name: str
            Name of the MAGIC calibrated file to use.
        is_mc: boolean
            Specify whether Monte Carlo (True) or data (False) events are read
        n_camera_pixels: int
            Number of MAGIC camera pixels (not hardcoded, but specified solely via ctapipe.instrument.CameraGeometry)

        Returns
        -------
        dict:
            A dictionary with the even properties: charge / arrival time data, trigger, direction etc.
        """

        event_data = dict()

        event_data['charge'] = []
        event_data['arrival_time'] = []
        event_data['trigger_pattern'] = np.array([], dtype=np.int32)
        event_data['stereo_event_number'] = np.array([], dtype=np.int32)
        event_data['pointing_zd'] = np.array([])
        event_data['pointing_az'] = np.array([])
        event_data['pointing_ra'] = np.array([])
        event_data['pointing_dec'] = np.array([])
        event_data['unix'] = np.array([])

        # monitoring information (updated from time to time)
        monitoring_data = dict()

        monitoring_data['badpixelinfo'] = []
        monitoring_data['badpixelinfoUnixRange'] = []
        monitoring_data['PedestalUnix'] = np.array([])
        monitoring_data['PedestalFundamental'] = dict()
        monitoring_data['PedestalFundamental']['Mean'] = []
        monitoring_data['PedestalFundamental']['Rms'] = []
        monitoring_data['PedestalFromExtractor'] = dict()
        monitoring_data['PedestalFromExtractor']['Mean'] = []
        monitoring_data['PedestalFromExtractor']['Rms'] = []
        monitoring_data['PedestalFromExtractorRndm'] = dict()
        monitoring_data['PedestalFromExtractorRndm']['Mean'] = []
        monitoring_data['PedestalFromExtractorRndm']['Rms'] = []

        event_data['file_edges'] = [0]

        # if no file in the list (e.g. when reading mono information), then simply
        # return empty dicts/array
        if uproot_file is None:
            return event_data, monitoring_data

        drive_data = dict()
        drive_data['mjd'] = np.array([])
        drive_data['zd'] = np.array([])
        drive_data['az'] = np.array([])
        drive_data['ra'] = np.array([])
        drive_data['dec'] = np.array([])

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

        input_file = uproot_file

        events = input_file['Events'].arrays(evt_common_list, library="np")

        # Reading the info common to MC and real data
        charge = events['MCerPhotEvt.fPixels.fPhot']
        arrival_time = events['MArrivalTime.fData']
        trigger_pattern = events['MTriggerPattern.fPrescaled']
        stereo_event_number = events['MRawEvtHeader.fStereoEvtNumber']

        if not is_mc:

            # Reading event timing information:
            event_times = input_file['Events'].arrays(time_array_list, library="np")
            
            # Computing the event arrival time
            event_obs_day = Time(event_times['MTime.fMjd'], format='mjd', scale='utc')
            event_obs_day = np.round(event_obs_day.to_value(format='unix', subfmt='float'))
            event_obs_day = np.array([Decimal(str(x)) for x in event_obs_day])

            event_millisec = np.round(event_times['MTime.fTime.fMilliSec'] * msec2sec, 3)
            event_millisec = np.array([Decimal(str(x)) for x in event_millisec])

            event_nanosec = np.round(event_times['MTime.fNanoSec'] * nsec2sec, 7)
            event_nanosec = np.array([Decimal(str(x)) for x in event_nanosec])

            event_unix = event_obs_day + event_millisec + event_nanosec
            event_data['unix'] = np.concatenate((event_data['unix'], event_unix))

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
            monitoring_data['badpixelinfo'].append(unsuitable_pix)
            # save time interval of badpixel info:
            monitoring_data['badpixelinfoUnixRange'].append([event_unix[0], event_unix[-1]])

        # try to read Pedestals tree (soft fail if not present)
            try:
                pedestal_info = input_file['Pedestals'].arrays(pedestal_array_list, library="np")

                pedestal_obs_day = Time(pedestal_info['MTimePedestals.fMjd'], format='mjd', scale='utc')
                pedestal_obs_day = np.round(pedestal_obs_day.to_value(format='unix', subfmt='float'))
                pedestal_obs_day = np.array([Decimal(str(x)) for x in pedestal_obs_day])

                pedestal_millisec = np.round(pedestal_info['MTimePedestals.fTime.fMilliSec'] * msec2sec, 3)
                pedestal_millisec = np.array([Decimal(str(x)) for x in pedestal_millisec])

                pedestal_nanosec = np.round(pedestal_info['MTimePedestals.fNanoSec'] * nsec2sec, 7)
                pedestal_nanosec = np.array([Decimal(str(x)) for x in pedestal_nanosec])

                pedestal_unix = pedestal_obs_day + pedestal_millisec + pedestal_nanosec
                monitoring_data['PedestalUnix']  = np.concatenate((monitoring_data['PedestalUnix'], pedestal_unix))

                n_pedestals = len(pedestal_unix)
                
                for quantity in ['Mean', 'Rms']:
                    for i_pedestal in range(n_pedestals):
                        monitoring_data['PedestalFundamental'][quantity].append(
                            pedestal_info[f'MPedPhotFundamental.fArray.f{quantity}'][i_pedestal][:n_camera_pixels])
                        monitoring_data['PedestalFromExtractor'][quantity].append(
                            pedestal_info[f'MPedPhotFromExtractor.fArray.f{quantity}'][i_pedestal][:n_camera_pixels])
                        monitoring_data['PedestalFromExtractorRndm'][quantity].append(
                            pedestal_info[f'MPedPhotFromExtractorRndm.fArray.f{quantity}'][i_pedestal][:n_camera_pixels])

            except KeyError:
                LOGGER.warning(
                    "Pedestals tree not present in file. Cleaning algorithm may fail.")

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

        # Reading pointing information (in units of degrees):
        if is_mc:
            # Retrieving the telescope pointing direction
            pointing = input_file['Events'].arrays(pointing_array_list, library="np")

            pointing_zd = pointing['MPointingPos.fZd'] - \
                pointing['MPointingPos.fDevZd']
            pointing_az = pointing['MPointingPos.fAz'] - \
                pointing['MPointingPos.fDevAz']
            # N.B. the positive sign here, as HA = local sidereal time - ra
            pointing_ra = (pointing['MPointingPos.fRa'] +
                           pointing['MPointingPos.fDevHa']) * degrees_per_hour
            pointing_dec = pointing['MPointingPos.fDec'] - \
                pointing['MPointingPos.fDevDec']

        # check for bit flips in the stereo event ID:
        d_x = np.diff(stereo_event_number.astype(np.int))
        dx_flip_ids_before = np.where(d_x < 0)[0]
        dx_flip_ids_after = dx_flip_ids_before + 1
        dx_flipzero_ids_first = np.where(d_x == 0)[0]
        dx_flipzero_ids_second = dx_flipzero_ids_first + 1
        if not is_mc:
            pedestal_ids = np.where(
                trigger_pattern == PEDESTAL_TRIGGER_PATTERN)[0]
            # sort out pedestals events from zero-difference steps:
            dx_flipzero_ids_second = np.array(
                list(set(dx_flipzero_ids_second) - set(pedestal_ids)))
            dx_flip_ids_after = np.array(np.union1d(
                dx_flip_ids_after, dx_flipzero_ids_second), dtype=np.int)
        else:
            # for MC, sort out stereo_event_number = 0:
            orphan_ids = np.where(stereo_event_number == 0)[0]
            dx_flip_ids_after = np.array(
                list(set(dx_flip_ids_after) - set(orphan_ids)))
        dx_flip_ids_before = dx_flip_ids_after - 1
        max_total_jumps = 100
        if len(dx_flip_ids_before) > 0:
            LOGGER.warning("Warning: detected %d bitflips in file %s. Flag affected events as unsuitable" % (
                len(dx_flip_ids_before), uproot_file.file_path))
            total_jumped_events = 0
            for i in dx_flip_ids_before:
                trigger_pattern[i] = -1
                trigger_pattern[i+1] = -1
                if not is_mc:
                    jumped_events = int(stereo_event_number[i]) - int(stereo_event_number[i+1])
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

        event_data['charge'].append(charge)
        event_data['arrival_time'].append(arrival_time)
        event_data['trigger_pattern'] = np.concatenate(
            (event_data['trigger_pattern'], trigger_pattern))
        event_data['stereo_event_number'] = np.concatenate(
            (event_data['stereo_event_number'], stereo_event_number))

        if is_mc:
            event_data['pointing_zd'] = np.concatenate(
                (event_data['pointing_zd'], pointing_zd))
            event_data['pointing_az'] = np.concatenate(
                (event_data['pointing_az'], pointing_az))
            event_data['pointing_ra'] = np.concatenate(
                (event_data['pointing_ra'], pointing_ra))
            event_data['pointing_dec'] = np.concatenate(
                (event_data['pointing_dec'], pointing_dec))

            mc_info = input_file['Events'].arrays(mc_list, library="np")
            # N.B.: For MC, there is only one subrun, so do not need to 'append'
            event_data['true_energy'] = mc_info['MMcEvt.fEnergy']
            event_data['true_zd'] = mc_info['MMcEvt.fTheta']
            event_data['true_az'] = mc_info['MMcEvt.fPhi']
            event_data['true_shower_primary_id'] = mc_info['MMcEvt.fPartId']
            event_data['true_h_first_int'] = mc_info['MMcEvt.fZFirstInteraction']
            event_data['true_core_x'] = mc_info['MMcEvt.fCoreX']
            event_data['true_core_y'] = mc_info['MMcEvt.fCoreY']

        event_data['file_edges'].append(len(event_data['trigger_pattern']))

        if not is_mc:
            monitoring_data['badpixelinfo'] = np.array(monitoring_data['badpixelinfo'])
            monitoring_data['badpixelinfoUnixRange'] = np.array(monitoring_data['badpixelinfoUnixRange'])
            # sort monitoring data:
            order = np.argsort(monitoring_data['PedestalUnix'])
            monitoring_data['PedestalUnix'] = monitoring_data['PedestalUnix'][order]

            for quantity in ['Mean', 'Rms']:
                monitoring_data['PedestalFundamental'][quantity] = np.array(
                    monitoring_data['PedestalFundamental'][quantity])
                monitoring_data['PedestalFromExtractor'][quantity] = np.array(
                    monitoring_data['PedestalFromExtractor'][quantity])
                monitoring_data['PedestalFromExtractorRndm'][quantity] = np.array(
                    monitoring_data['PedestalFromExtractorRndm'][quantity])

            # get only drive reports with unique times, otherwise interpolation fails.
            drive_mjd_unique, unique_indices = np.unique(
                drive_data['mjd'],
                return_index=True
            )
            drive_zd_unique = drive_data['zd'][unique_indices]
            drive_az_unique = drive_data['az'][unique_indices]
            drive_ra_unique = drive_data['ra'][unique_indices]
            drive_dec_unique = drive_data['dec'][unique_indices]

            first_drive_report_time = Time(drive_mjd_unique[0], scale='utc', format='mjd')
            last_drive_report_time = Time(drive_mjd_unique[-1], scale='utc', format='mjd')

            LOGGER.warning(f"Interpolating events information from {len(drive_data['mjd'])} drive reports.")
            LOGGER.warning(f"Drive reports available from {first_drive_report_time.iso} to {last_drive_report_time.iso}.")

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

            # Interpolating the drive pointing to the event time stamps
            event_mjd = Time(event_data['unix'], format='unix', scale='utc').to_value(format='mjd', subfmt='float')

            event_data['pointing_zd'] = drive_zd_pointing_interpolator(event_mjd)
            event_data['pointing_az'] = drive_az_pointing_interpolator(event_mjd)
            event_data['pointing_ra'] = drive_ra_pointing_interpolator(event_mjd)
            event_data['pointing_dec'] = drive_dec_pointing_interpolator(event_mjd)

        return event_data, monitoring_data

    def _find_pedestal_events(self):
        """
        This internal method identifies the IDs (order numbers) of the
        pedestal events in the run.

        Returns
        -------
        dict:
            A dictionary of pedestal event IDs in M1/2 separately.
        """

        pedestal_ids = dict()

        for telescope in self.event_data:
            ped_triggers = np.where(
                self.event_data[telescope]['trigger_pattern'] == PEDESTAL_TRIGGER_PATTERN)
            pedestal_ids[telescope] = ped_triggers[0]

        return pedestal_ids

    def _find_stereo_events(self):
        """
        This internal methods identifies stereo events in the run.

        Returns
        -------
        list:
            A list of pairs (M1_id, M2_id) corresponding to stereo events in the run.
        """

        stereo_ids = []

        n_m1_events = len(self.event_data['M1']['stereo_event_number'])
        n_m2_events = len(self.event_data['M2']['stereo_event_number'])
        if (n_m1_events == 0) or (n_m2_events == 0):
            return stereo_ids

        if not self.is_mc:
            stereo_m1_data = self.event_data['M1']['stereo_event_number'][np.where(self.event_data['M1']['trigger_pattern'] == DATA_STEREO_TRIGGER_PATTERN)]
            stereo_m2_data = self.event_data['M2']['stereo_event_number'][np.where(self.event_data['M2']['trigger_pattern'] == DATA_STEREO_TRIGGER_PATTERN)]

            # find common values between M1 and M2 stereo events, see https://numpy.org/doc/stable/reference/generated/numpy.intersect1d.html
            stereo_numbers = np.intersect1d(stereo_m1_data, stereo_m2_data)

            # find indices of the stereo event numbers in original stereo event numbers arrays, see
            # https://stackoverflow.com/questions/12122639/find-indices-of-a-list-of-values-in-a-numpy-array
            m1_ids = np.searchsorted(self.event_data['M1']['stereo_event_number'], stereo_numbers)
            m2_ids = np.searchsorted(self.event_data['M2']['stereo_event_number'], stereo_numbers)

            # make list of tuples, see https://stackoverflow.com/questions/2407398/how-to-merge-lists-into-a-list-of-tuples
            stereo_ids = list(zip(m1_ids, m2_ids))
        else:
            stereo_m1_data = self.event_data['M1']['stereo_event_number'][np.where(self.event_data['M1']['trigger_pattern'] == MC_STEREO_TRIGGER_PATTERN)]
            stereo_m2_data = self.event_data['M2']['stereo_event_number'][np.where(self.event_data['M2']['trigger_pattern'] == MC_STEREO_TRIGGER_PATTERN)]
            # remove events with 0 stereo number, which are mono events
            stereo_m1_data = stereo_m1_data[np.where(stereo_m1_data != 0)]
            stereo_m2_data = stereo_m2_data[np.where(stereo_m2_data != 0)]

            stereo_numbers = np.intersect1d(stereo_m1_data, stereo_m2_data)

            # because of IDs equal to 0, we must find indices in a slight different way
            # see https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array

            index_m1 = np.argsort(self.event_data['M1']['stereo_event_number'])
            index_m2 = np.argsort(self.event_data['M2']['stereo_event_number'])

            sort_stereo_events_m1 = self.event_data['M1']['stereo_event_number'][index_m1]
            sort_stereo_events_m2 = self.event_data['M2']['stereo_event_number'][index_m2]

            sort_index_m1 = np.searchsorted(sort_stereo_events_m1, stereo_numbers)
            sort_index_m2 = np.searchsorted(sort_stereo_events_m2, stereo_numbers)

            m1_ids = np.take(index_m1, sort_index_m1)
            m2_ids = np.take(index_m2, sort_index_m2)

            stereo_ids = list(zip(m1_ids, m2_ids))

        return stereo_ids

    def _find_stereo_mc_events(self):
        """
        This internal methods identifies stereo events in the run.

        Returns
        -------
        list:
            A list of pairs (M1_id, M2_id) corresponding to stereo events in the run.
        """

        mono_ids = dict()
        mono_ids['M1'] = []
        mono_ids['M2'] = []

        m1_ids = np.argwhere(self.event_data['M1']['stereo_event_number'])
        m2_ids = np.argwhere(self.event_data['M2']['stereo_event_number'])

        mono_ids['M1'] = list(m1_ids.flatten())
        mono_ids['M2'] = list(m2_ids.flatten())

        return mono_ids

    def _find_mono_events(self):
        """
        This internal method identifies the IDs (order numbers) of the
        pedestal events in the run.

        Returns
        -------
        dict:
            A dictionary of pedestal event IDs in M1/2 separately.
        """

        mono_ids = dict()
        mono_ids['M1'] = []
        mono_ids['M2'] = []

        n_m1_events = len(self.event_data['M1']['stereo_event_number'])
        n_m2_events = len(self.event_data['M2']['stereo_event_number'])

        if not self.is_mc:
            if (n_m1_events != 0) and (n_m2_events != 0):
                m1_data = self.event_data['M1']['stereo_event_number'][np.where(self.event_data['M1']['trigger_pattern'] == DATA_STEREO_TRIGGER_PATTERN)]
                m2_data = self.event_data['M2']['stereo_event_number'][np.where(self.event_data['M2']['trigger_pattern'] == DATA_STEREO_TRIGGER_PATTERN)]

                m1_ids_data = np.where(self.event_data['M1']['trigger_pattern'] == DATA_STEREO_TRIGGER_PATTERN)[0]
                m2_ids_data = np.where(self.event_data['M2']['trigger_pattern'] == DATA_STEREO_TRIGGER_PATTERN)[0]

                stereo_numbers = np.intersect1d(m1_data, m2_data)

                m1_ids_stereo = np.searchsorted(self.event_data['M1']['stereo_event_number'], stereo_numbers)
                m2_ids_stereo = np.searchsorted(self.event_data['M2']['stereo_event_number'], stereo_numbers)

                # remove ids that have stereo trigger from the array of ids of data events
                # see: https://stackoverflow.com/questions/52417929/remove-elements-from-one-array-if-present-in-another-array-keep-duplicates-nu

                sidx1 = m1_ids_stereo.argsort()
                idx1 = np.searchsorted(m1_ids_stereo,m1_ids_data,sorter=sidx1)
                idx1[idx1==len(m1_ids_stereo)] = 0
                m1_ids_mono = m1_ids_data[m1_ids_stereo[sidx1[idx1]] != m1_ids_data]

                sidx2 = m2_ids_stereo.argsort()
                idx2 = np.searchsorted(m2_ids_stereo,m2_ids_data,sorter=sidx2)
                idx2[idx2==len(m2_ids_stereo)] = 0
                m2_ids_mono = m2_ids_data[m2_ids_stereo[sidx2[idx2]] != m2_ids_data]

                mono_ids['M1'] = m1_ids_mono.tolist()
                mono_ids['M2'] = m2_ids_mono.tolist()
            elif (n_m1_events != 0) and (n_m2_events == 0):
                m1_ids_data = np.where(self.event_data['M1']['trigger_pattern'] == DATA_STEREO_TRIGGER_PATTERN)[0]
                mono_ids['M1'] = m1_ids_data.tolist()
            elif (n_m1_events == 0) and (n_m2_events != 0):
                m2_ids_data = np.where(self.event_data['M2']['trigger_pattern'] == DATA_STEREO_TRIGGER_PATTERN)[0]
                mono_ids['M2'] = m2_ids_data.tolist()
        else:
            # just find ids where event stereo number is 0 (which is given to mono events) and pattern is MC trigger
            m1_mono_mask = np.logical_and(self.event_data['M1']['trigger_pattern'] == MC_STEREO_TRIGGER_PATTERN, self.event_data['M1']['stereo_event_number'] == 0)
            m2_mono_mask = np.logical_and(self.event_data['M2']['trigger_pattern'] == MC_STEREO_TRIGGER_PATTERN, self.event_data['M2']['stereo_event_number'] == 0)

            m1_ids = np.where(m1_mono_mask == True)[0].tolist()
            m2_ids = np.where(m2_mono_mask == True)[0].tolist()

            mono_ids['M1'] = m1_ids
            mono_ids['M2'] = m2_ids

        return mono_ids

    def _get_pedestal_file_num(self, pedestal_event_num, telescope):
        """
        This internal method identifies the M1/2 file number of the
        given pedestal event in M1/2 file lists, corresponding to this run.

        Parameters
        ----------
        pedestal_event_num: int
            Order number of the event in the list of pedestal events
            of the specified telescope, corresponding to this run.
        telescope: str
            The name of the telescope to which this event corresponds.
            May be "M1" or "M2".

        Returns
        -------
        file_num:
            Order number of the corresponding file in the M1 or M2 file list.
        """

        event_id = self.pedestal_ids[telescope][pedestal_event_num]
        file_num = np.digitize(
            [event_id], self.event_data[telescope]['file_edges'])
        file_num = file_num[0] - 1

        return file_num

    def _get_stereo_file_num(self, stereo_event_num):
        """
        This internal method identifies the M1/2 file numbers of the
        given stereo event in M1/2 file lists, corresponding to this run.

        Parameters
        ----------
        stereo_event_num: int
            Order number of the event in the list of stereo events corresponding
            to this run.

        Returns
        -------
        m1_file_num:
            Order number of the corresponding file in the M1 file list.
        m2_file_num:
            Order number of the corresponding file in the M2 file list.
        """

        m1_id = self.stereo_ids[stereo_event_num][0]
        m2_id = self.stereo_ids[stereo_event_num][1]
        m1_file_num = np.digitize([m1_id], self.event_data['M1']['file_edges'])
        m2_file_num = np.digitize([m2_id], self.event_data['M2']['file_edges'])

        m1_file_num = m1_file_num[0] - 1
        m2_file_num = m2_file_num[0] - 1

        return m1_file_num, m2_file_num

    def _get_mono_file_num(self, mono_event_num, telescope):
        """
        This internal method identifies the M1/2 file number of the
        given mono event in M1/2 file lists, corresponding to this run.

        Parameters
        ----------
        mono_event_num: int
            Order number of the event in the list of stereo events corresponding
            to this run.
        telescope: str
            The name of the telescope to which this event corresponds.
            May be "M1" or "M2".

        Returns
        -------
        file_num:
            Order number of the corresponding file in the M1 or M2 file list.
        """

        event_id = self.mono_ids[telescope][mono_event_num]
        file_num = np.digitize(
            [event_id], self.event_data[telescope]['file_edges'])
        file_num = file_num[0] - 1

        return file_num

    def get_pedestal_event_data(self, pedestal_event_num, telescope):
        """
        This method read the photon content and arrival time (per pixel)
        for the specified pedestal event. Also returned is the event telescope pointing
        data.

        Parameters
        ----------
        pedestal_event_num: int
            Order number of the event in the list of pedestal events for the
            given telescope, corresponding to this run.
        telescope: str
            The name of the telescope to which this event corresponds.
            May be "M1" or "M2".

        Returns
        -------
        dict:
            The output has the following structure:
            'image' - photon_content in requested telescope
            'pulse_time' - arrival_times in requested telescope
            'pointing_az' - pointing azimuth [degrees]
            'pointing_zd' - pointing zenith angle [degrees]
            'pointing_ra' - pointing right ascension [degrees]
            'pointing_dec' - pointing declination [degrees]
            'unix' - event arrival time [unix]
        """

        file_num = self._get_pedestal_file_num(pedestal_event_num, telescope)
        event_id = self.pedestal_ids[telescope][pedestal_event_num]

        id_in_file = event_id - self.event_data[telescope]['file_edges'][file_num]

        photon_content = self.event_data[telescope]['charge'][file_num][id_in_file][:self.n_camera_pixels]
        arrival_times = self.event_data[telescope]['arrival_time'][file_num][id_in_file][:self.n_camera_pixels]

        event_data = dict()
        event_data['image'] = np.array(photon_content)
        event_data['pulse_time'] = np.array(arrival_times)
        event_data['pointing_az'] = self.event_data[telescope]['pointing_az'][event_id]
        event_data['pointing_zd'] = self.event_data[telescope]['pointing_zd'][event_id]
        event_data['pointing_ra'] = self.event_data[telescope]['pointing_ra'][event_id]
        event_data['pointing_dec'] = self.event_data[telescope]['pointing_dec'][event_id]
        event_data['unix'] = self.event_data[telescope]['unix'][event_id]

        return event_data

    def get_stereo_event_data(self, stereo_event_num):
        """
        This method read the photon content and arrival time (per pixel)
        for the specified stereo event. Also returned is the event telescope pointing
        data.

        Parameters
        ----------
        stereo_event_num: int
            Order number of the event in the list of stereo events corresponding
            to this run.

        Returns
        -------
        dict:
            The output has the following structure:
            'm1_image' - M1 photon_content
            'm1_pulse_time' - M1 arrival_times
            'm2_image' - M2 photon_content
            'm2_peak_pos' - M2 arrival_times
            'm1_pointing_az' - M1 pointing azimuth [degrees]
            'm1_pointing_zd' - M1 pointing zenith angle [degrees]
            'm1_pointing_ra' - M1 pointing right ascension [degrees]
            'm1_pointing_dec' - M1 pointing declination [degrees]
            'm2_pointing_az' - M2 pointing azimuth [degrees]
            'm2_pointing_zd' - M2 pointing zenith angle [degrees]
            'm2_pointing_ra' - M2 pointing right ascension [degrees]
            'm2_pointing_dec' - M2 pointing declination [degrees]
            'm1_unix' - M1 event arrival time [unix]
            'm2_unix' - M2 event arrival time [unix]
        """

        m1_file_num, m2_file_num = self._get_stereo_file_num(stereo_event_num)
        m1_id = self.stereo_ids[stereo_event_num][0]
        m2_id = self.stereo_ids[stereo_event_num][1]

        m1_id_in_file = m1_id - \
            self.event_data['M1']['file_edges'][m1_file_num]
        m2_id_in_file = m2_id - \
            self.event_data['M2']['file_edges'][m2_file_num]

        m1_photon_content = self.event_data['M1']['charge'][m1_file_num][m1_id_in_file][:self.n_camera_pixels]
        m1_arrival_times = self.event_data['M1']['arrival_time'][m1_file_num][m1_id_in_file][:self.n_camera_pixels]

        m2_photon_content = self.event_data['M2']['charge'][m2_file_num][m2_id_in_file][:self.n_camera_pixels]
        m2_arrival_times = self.event_data['M2']['arrival_time'][m2_file_num][m2_id_in_file][:self.n_camera_pixels]

        event_data = dict()
        event_data['m1_image'] = np.array(m1_photon_content)
        event_data['m1_pulse_time'] = np.array(m1_arrival_times)
        event_data['m2_image'] = np.array(m2_photon_content)
        event_data['m2_pulse_time'] = np.array(m2_arrival_times)
        event_data['m1_pointing_az'] = self.event_data['M1']['pointing_az'][m1_id]
        event_data['m1_pointing_zd'] = self.event_data['M1']['pointing_zd'][m1_id]
        event_data['m1_pointing_ra'] = self.event_data['M1']['pointing_ra'][m1_id]
        event_data['m1_pointing_dec'] = self.event_data['M1']['pointing_dec'][m1_id]
        event_data['m2_pointing_az'] = self.event_data['M2']['pointing_az'][m2_id]
        event_data['m2_pointing_zd'] = self.event_data['M2']['pointing_zd'][m2_id]
        event_data['m2_pointing_ra'] = self.event_data['M2']['pointing_ra'][m2_id]
        event_data['m2_pointing_dec'] = self.event_data['M2']['pointing_dec'][m2_id]

        if not self.is_mc:
            event_data['m1_unix'] = self.event_data['M1']['unix'][m1_id]
            event_data['m2_unix'] = self.event_data['M2']['unix'][m2_id]

        else:
            event_data['true_energy'] = self.event_data['M1']['true_energy'][m1_id]
            event_data['true_zd'] = self.event_data['M1']['true_zd'][m1_id]
            event_data['true_az'] = self.event_data['M1']['true_az'][m1_id]
            event_data['true_shower_primary_id'] = self.event_data['M1']['true_shower_primary_id'][m1_id]
            event_data['true_h_first_int'] = self.event_data['M1']['true_h_first_int'][m1_id]
            event_data['true_core_x'] = self.event_data['M1']['true_core_x'][m1_id]
            event_data['true_core_y'] = self.event_data['M1']['true_core_y'][m1_id]

        return event_data

    def get_mono_event_data(self, mono_event_num, telescope):
        """
        This method read the photon content and arrival time (per pixel)
        for the specified mono event. Also returned is the event telescope pointing
        data.

        Parameters
        ----------
        mono_event_num: int
            Order number of the event in the list of mono events for the
            given telescope, corresponding to this run.
        telescope: str
            The name of the telescope to which this event corresponds.
            May be "M1" or "M2".

        Returns
        -------
        dict:
            The output has the following structure:
            'image' - photon_content in requested telescope
            'pulse_time' - arrival_times in requested telescope
            'pointing_az' - pointing azimuth [degrees]
            'pointing_zd' - pointing zenith angle [degrees]
            'pointing_ra' - pointing right ascension [degrees]
            'pointing_dec' - pointing declination [degrees]
            'unix' - event arrival time [unix]
        """

        file_num = self._get_mono_file_num(mono_event_num, telescope)
        event_id = self.mono_ids[telescope][mono_event_num]

        id_in_file = event_id - \
            self.event_data[telescope]['file_edges'][file_num]

        photon_content = self.event_data[telescope]['charge'][file_num][id_in_file][:self.n_camera_pixels]
        arrival_times = self.event_data[telescope]['arrival_time'][file_num][id_in_file][:self.n_camera_pixels]

        event_data = dict()
        event_data['image'] = np.array(photon_content, dtype=np.float)
        event_data['pulse_time'] = np.array(arrival_times, dtype=np.float)
        event_data['pointing_az'] = self.event_data[telescope]['pointing_az'][event_id]
        event_data['pointing_zd'] = self.event_data[telescope]['pointing_zd'][event_id]
        event_data['pointing_ra'] = self.event_data[telescope]['pointing_ra'][event_id]
        event_data['pointing_dec'] = self.event_data[telescope]['pointing_dec'][event_id]

        if not self.is_mc:
            event_data['unix'] = self.event_data[telescope]['unix'][event_id]

        else:
            event_data['true_energy'] = self.event_data[telescope]['true_energy'][event_id]
            event_data['true_zd'] = self.event_data[telescope]['true_zd'][event_id]
            event_data['true_az'] = self.event_data[telescope]['true_az'][event_id]
            event_data['true_shower_primary_id'] = self.event_data[telescope]['true_shower_primary_id'][event_id]
            event_data['true_h_first_int'] = self.event_data[telescope]['true_h_first_int'][event_id]
            event_data['true_core_x'] = self.event_data[telescope]['true_core_x'][event_id]
            event_data['true_core_y'] = self.event_data[telescope]['true_core_y'][event_id]

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
