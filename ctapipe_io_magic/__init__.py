"""
# Event source for MAGIC calibrated data files.
# Requires uproot package (https://github.com/scikit-hep/uproot).
"""

import os
import re
import scipy
import scipy.interpolate
import uproot
import logging
import numpy as np
from pathlib import Path
from decimal import Decimal
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import Angle
from pkg_resources import resource_filename

from ctapipe.io import EventSource, DataLevel
from ctapipe.core import Container, Field, Provenance
from ctapipe.core.traits import Bool
from ctapipe.coordinates import CameraFrame

from ctapipe.containers import (
    EventType,
    ArrayEventContainer,
    SimulationConfigContainer,
    SimulatedEventContainer,
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
    DATA_STEREO_TRIGGER_PATTERN,
)

__all__ = ['MAGICEventSource', 'MARSDataLevel', '__version__']

logger = logging.getLogger(__name__)

degrees_per_hour = 15.0
seconds_per_hour = 3600.

msec2sec = 1e-3
nsec2sec = 1e-9

mc_data_type = 256

MAGIC_TO_CTA_EVENT_TYPE = {
    MC_STEREO_TRIGGER_PATTERN: EventType.SUBARRAY,
    DATA_STEREO_TRIGGER_PATTERN: EventType.SUBARRAY,
    PEDESTAL_TRIGGER_PATTERN: EventType.SKY_PEDESTAL,
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
    generate_pedestals : bool
        Flag indicating if pedestal events should be returned by the generator
    """

    process_run = Bool(
        default_value=True,
        help='Read all subruns from a given run.'
    ).tag(config=True)

    generate_pedestals = Bool(
           default_value=False,
           help='Extract pedestal events instead of cosmic events.'
    ).tag(config=True)

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

        super().__init__(
            input_url=input_url,
            config=config,
            parent=parent,
            **kwargs
        )

        path, name = os.path.split(os.path.abspath(self.input_url))
        info_tuple = self.get_run_info_from_name(name)
        run = info_tuple[0]
        telescope = info_tuple[2]

        regex = rf"\d{{8}}_M{telescope}_0{run}\.\d{{3}}_Y_.*\.root"
        regex_mc = rf"GA_M{telescope}_\w+_{run}_Y_.*\.root"

        reg_comp = re.compile(regex)
        reg_comp_mc = re.compile(regex_mc)

        ls = Path(path).iterdir()
        self.file_list_drive = []

        for file_path in ls:
            if reg_comp.match(file_path.name) is not None or reg_comp_mc.match(file_path.name) is not None:
                self.file_list_drive.append(file_path)

        self.file_list_drive.sort()

        if self.process_run:
            self.file_list = self.file_list_drive
        else:
            self.file_list = [self.input_url]

        # Retrieving the list of run numbers corresponding to the data files
        self.files_ = [uproot.open(rootf) for rootf in self.file_list]
        run_info = self.parse_run_info()

        self.run_numbers = run_info[0]
        self.is_mc = run_info[1][0]
        self.telescope = run_info[2][0]
        self.mars_datalevel = run_info[3][0]

        self.metadata = self.parse_metadata_info()

        # Retrieving the data level (so far HARDCODED Sorcerer)
        self.datalevel = DataLevel.DL0

        if self.is_simulation:
            self.simulation_config = self.parse_simulation_header()

        if not self.is_simulation:
            self.is_stereo, self.is_sumt = self.parse_data_info()

        # # Setting up the current run with the first run present in the data
        # self.current_run = self._set_active_run(run_number=0)
        self.current_run = None

        self._subarray_info = self.prepare_subarray_info()

        if not self.is_simulation:
            self.drive_information = self.prepare_drive_information()

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

        for rootf in self.files_:
            rootf.close()

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

        file_name = str(file_name)
        mask_data_calibrated = r"\d{8}_M(\d+)_(\d+)\.\d+_Y_.*"
        mask_data_star = r"\d{8}_M(\d+)_(\d+)\.\d+_I_.*"
        mask_data_superstar = r"\d{8}_(\d+)_S_.*"
        mask_data_melibea = r"\d{8}_(\d+)_Q_.*"
        mask_mc_calibrated = r"GA_M(\d)_za\d+to\d+_\d_(\d+)_Y_.*"
        mask_mc_star = r"GA_M(\d)_za\d+to\d+_\d_(\d+)_I_.*"
        mask_mc_superstar = r"GA_za\d+to\d+_\d_S_.*"
        mask_mc_melibea = r"GA_za\d+to\d+_\d_Q_.*"
        if re.match(mask_data_calibrated, file_name) is not None:
            parsed_info = re.match(mask_data_calibrated, file_name)
            telescope = int(parsed_info.group(1))
            run_number = int(parsed_info.group(2))
            datalevel = MARSDataLevel.CALIBRATED
            is_mc = False
        elif re.match(mask_data_star, file_name) is not None:
            parsed_info = re.match(mask_data_star, file_name)
            telescope = int(parsed_info.group(1))
            run_number = int(parsed_info.group(2))
            datalevel = MARSDataLevel.STAR
            is_mc = False
        elif re.match(mask_data_superstar, file_name) is not None:
            parsed_info = re.match(mask_data_superstar, file_name)
            telescope = None
            run_number = int(parsed_info.grou(1))
            datalevel = MARSDataLevel.SUPERSTAR
            is_mc = False
        elif re.match(mask_data_melibea, file_name) is not None:
            parsed_info = re.match(mask_data_melibea, file_name)
            telescope = None
            run_number = int(parsed_info.grou(1))
            datalevel = MARSDataLevel.MELIBEA
            is_mc = False
        elif re.match(mask_mc_calibrated, file_name) is not None:
            parsed_info = re.match(mask_mc_calibrated, file_name)
            telescope = int(parsed_info.group(1))
            run_number = int(parsed_info.group(2))
            datalevel = MARSDataLevel.CALIBRATED
            is_mc = True
        elif re.match(mask_mc_star, file_name) is not None:
            parsed_info = re.match(mask_mc_star, file_name)
            telescope = int(parsed_info.group(1))
            run_number = int(parsed_info.group(2))
            datalevel = MARSDataLevel.STAR
            is_mc = True
        elif re.match(mask_mc_superstar, file_name) is not None:
            parsed_info = re.match(mask_mc_superstar, file_name)
            telescope = None
            run_number = None
            datalevel = MARSDataLevel.SUPERSTAR
            is_mc = True
        elif re.match(mask_mc_melibea, file_name) is not None:
            parsed_info = re.match(mask_mc_melibea, file_name)
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

        run_numbers = []
        is_simulation = []
        telescope_numbers = []
        datalevels = []

        for rootf in self.files_:

            run_info = rootf['RunHeaders'].arrays(
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

            if run_type == mc_data_type:
                is_mc = True
            else:
                is_mc = False

            events_tree = rootf['Events']

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

            run_numbers.append(run_number)
            is_simulation.append(is_mc)
            telescope_numbers.append(telescope_number)
            datalevels.append(datalevel)

        run_numbers = np.unique(run_numbers).tolist()
        is_simulation = np.unique(is_simulation).tolist()
        telescope_numbers = np.unique(telescope_numbers).tolist()
        datalevels = np.unique(datalevels).tolist()

        if len(is_simulation) > 1:
            raise ValueError(
                "Loaded files contain both real data and simulation runs. \
                 Please load only data OR Monte Carlos.")
        if len(telescope_numbers) > 1:
            raise ValueError(
                "Loaded files contain data from different telescopes. \
                 Please load data belonging to the same telescope.")
        if len(datalevels) > 1:
            raise ValueError(
                "Loaded files contain data at different datalevels. \
                 Please load data belonging to the same datalevel.")

        return run_numbers, is_simulation, telescope_numbers, datalevels

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

        is_stereo = []
        is_sumt = []

        for rootf in self.files_:

            trigger_tree = rootf["Trigger"]
            L3T_tree = rootf["L3T"]

            # here we take the 2nd element (if possible) because sometimes
            # the first trigger report has still the old prescaler values from a previous run
            try:
                prescaler_array = trigger_tree["MTriggerPrescFact.fPrescFact"].array(library="np")
            except AssertionError:
                LOGGER.warning("No prescaler info found. Will assume standard stereo data.")
                stereo = True
                sumt = False
                return stereo, sumt

            prescaler_size = prescaler_array.size
            if prescaler_size > 1:
                prescaler = prescaler_array[1]
            else:
                prescaler = prescaler_array[0]

            if prescaler == prescaler_mono_nosumt or prescaler == prescaler_mono_sumt:
                stereo = False
            elif prescaler == prescaler_stereo:
                stereo = True
            else:
                stereo = True

            sumt = False
            if stereo:
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
                    sumt = True
                elif L3Table == L3_table_nosumt:
                    sumt = False
                else:
                    sumt = False
            else:
                if prescaler == prescaler_mono_sumt:
                    sumt = True

            is_stereo.append(stereo)
            is_sumt.append(sumt)

        is_stereo = np.unique(is_stereo).tolist()
        is_sumt = np.unique(is_sumt).tolist()

        if len(is_stereo) > 1:
            raise ValueError(
                "Loaded files contain both stereo and mono data. \
                 Please load only stereo or mono.")

        if len(is_sumt) > 1:
            LOGGER.warning("Found data with both standard and Sum trigger. While this is \
                not an issue, check that this is what you really want to do.")

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
            subarray = subarray.select_subarray(self.allowed_tels)

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
        metadata["file_list"] = self.file_list
        metadata['run_numbers'] = self.run_numbers
        metadata['is_simulation'] = self.is_simulation
        metadata['telescope'] = self.telescope
        metadata['subrun_number'] = []
        metadata['source_ra'] = []
        metadata['source_dec'] = []
        metadata['source_name'] = []
        metadata['observation_mode'] = []
        metadata['mars_version_sorcerer'] = []
        metadata['root_version_sorcerer'] = []

        for rootf in self.files_:

            meta_info_runh = rootf['RunHeaders'].arrays(
                    metadatainfo_array_list_runheaders, library="np"
            )

            metadata['subrun_number'].append(int(meta_info_runh['MRawRunHeader.fSubRunIndex'][0]))
            metadata['source_ra'].append(
                meta_info_runh['MRawRunHeader.fSourceRA'][0] / seconds_per_hour * degrees_per_hour * u.deg
            )
            metadata['source_dec'].append(
                meta_info_runh['MRawRunHeader.fSourceDEC'][0] / seconds_per_hour * u.deg
            )
            if not self.is_simulation:
                src_name_array = meta_info_runh['MRawRunHeader.fSourceName[80]'][0]
                metadata['source_name'].append("".join([chr(item) for item in src_name_array if item != 0]))
                obs_mode_array = meta_info_runh['MRawRunHeader.fObservationMode[60]'][0]
                metadata['observation_mode'].append("".join([chr(item) for item in obs_mode_array if item != 0]))

            meta_info_runt = rootf['RunTails'].arrays(
                metadatainfo_array_list_runtails,
                library="np"
            )

            mars_version_encoded = int(meta_info_runt['MMarsVersion_sorcerer.fMARSVersionCode'][0])
            root_version_encoded = int(meta_info_runt['MMarsVersion_sorcerer.fROOTVersionCode'][0])
            metadata['mars_version_sorcerer'].append(self.decode_version_number(mars_version_encoded))
            metadata['root_version_sorcerer'].append(self.decode_version_number(root_version_encoded))

        return metadata

    def parse_simulation_header(self):
        """
        Parse the simulation information from the RunHeaders tree.
        The MMcCorsikaRunHeader and MMcRunHeader branches are used.

        Returns
        -------
        simulation_config: ctapipe.containers.SimulationConfigContainer
            Container filled with simulation information
        """

        mc_header_branches = [
            'MMcCorsikaRunHeader.fCorsikaVersion',
            'MMcCorsikaRunHeader.fELowLim',
            'MMcCorsikaRunHeader.fEUppLim',
            'MMcCorsikaRunHeader.fSlopeSpec',
            'MMcCorsikaRunHeader.fHeightLev[10]',
            'MMcCorsikaRunHeader.fAtmosphericModel',
            'MMcRunHeader.fRandomPointingConeSemiAngle',
            'MMcRunHeader.fImpactMax',
            'MMcRunHeader.fNumSimulatedShowers',
            'MMcRunHeader.fShowerThetaMax',
            'MMcRunHeader.fShowerThetaMin',
            'MMcRunHeader.fShowerPhiMax',
            'MMcRunHeader.fShowerPhiMin',
            'MMcRunHeader.fCWaveUpper',
            'MMcRunHeader.fCWaveLower'
        ]

        # Magnetic field values at the MAGIC site (taken from CORSIKA input cards)
        # Reference system is the CORSIKA one, where x-axis points to magnetic north, i.e., B y-component is 0
        # mfield_dec is the magnetic declination i.e. angle between magnetic and
        # geographic north, negative if pointing westwards, positive if pointing eastwards
        # mfield_inc is the magnetic field inclination

        mfield_x = u.Quantity(29.5, u.uT)
        mfield_z = u.Quantity(-23.0, u.uT)
        mfield_total = np.sqrt(mfield_x ** 2 + mfield_z ** 2)
        mfield_dec = u.Quantity(-7.0, u.deg)
        mfield_inc = np.arctan2(mfield_z, mfield_x)

        shower_prog_id = 1   # CORSIKA=1, ALTAI=2, KASCADE=3, MOCCA=4
        shower_reuse = 5     # every shower reused 5 times for standard MAGIC MC
        is_diffuse = True

        uproot_file = self.files_[0]

        sim_info = uproot_file['RunHeaders'].arrays(
            expressions=mc_header_branches,
            library='np'
        )

        if np.isclose(sim_info['MMcRunHeader.fRandomPointingConeSemiAngle'][0], 0.4):
            max_viewcone_radius = u.Quantity(0.4, u.deg)
            min_viewcone_radius = u.Quantity(0.4, u.deg)

        else:
            max_viewcone_radius = u.Quantity(sim_info['MMcRunHeader.fRandomPointingConeSemiAngle'][0], u.deg)
            min_viewcone_radius = u.Quantity(0, u.deg)

        simulation_config = SimulationConfigContainer(
            corsika_version=sim_info['MMcCorsikaRunHeader.fCorsikaVersion'][0],
            energy_range_min=u.Quantity(sim_info['MMcCorsikaRunHeader.fELowLim'][0], u.GeV).to(u.TeV),
            energy_range_max=u.Quantity(sim_info['MMcCorsikaRunHeader.fEUppLim'][0], u.GeV).to(u.TeV),
            prod_site_B_total=mfield_total,
            prod_site_B_declination=mfield_dec.to(u.rad),
            prod_site_B_inclination=mfield_inc.to(u.rad),
            prod_site_alt=u.Quantity(sim_info['MMcCorsikaRunHeader.fHeightLev[10]'][0][0], u.cm).to(u.m),
            spectral_index=sim_info['MMcCorsikaRunHeader.fSlopeSpec'][0],
            shower_prog_id=shower_prog_id,
            num_showers=sim_info['MMcRunHeader.fNumSimulatedShowers'][0],
            shower_reuse=shower_reuse,
            max_alt=u.Quantity(90 - sim_info['MMcRunHeader.fShowerThetaMax'][0], u.deg).to(u.rad),
            min_alt=u.Quantity(90 - sim_info['MMcRunHeader.fShowerThetaMin'][0], u.deg).to(u.rad),
            max_az=u.Quantity(sim_info['MMcRunHeader.fShowerPhiMax'][0], u.deg).to(u.rad),
            min_az=u.Quantity(sim_info['MMcRunHeader.fShowerPhiMin'][0], u.deg).to(u.rad),
            diffuse=is_diffuse,
            max_viewcone_radius=max_viewcone_radius,
            min_viewcone_radius=min_viewcone_radius,
            max_scatter_range=u.Quantity(sim_info['MMcRunHeader.fImpactMax'][0], u.cm).to(u.m),
            min_scatter_range=u.Quantity(0, u.m),
            atmosphere=sim_info['MMcCorsikaRunHeader.fAtmosphericModel'][0],
            corsika_wlen_min=u.Quantity(sim_info['MMcRunHeader.fCWaveLower'][0], u.nm),
            corsika_wlen_max=u.Quantity(sim_info['MMcRunHeader.fCWaveUpper'][0], u.nm)
        )

        return simulation_config

    def prepare_drive_information(self):

        drive_leaves = {
            'mjd': 'MReportDrive.fMjd',
            'zd': 'MReportDrive.fCurrentZd',
            'az': 'MReportDrive.fCurrentAz',
            'ra': 'MReportDrive.fRa',
            'dec': 'MReportDrive.fDec',
        }

        # Getting the telescope drive info
        drive_data = {k: [] for k in drive_leaves.keys()}

        if self.process_run:
            rootfiles = self.files_
        else:
            rootfiles = [uproot.open(rootf) for rootf in self.file_list_drive]

        for rootf in rootfiles:
            drive = rootf["Drive"].arrays(drive_leaves.values(), library="np")

            n_reports = len(drive['MReportDrive.fMjd'])
            if n_reports == 0:
                raise MissingDriveReportError(f"File {rootf.file_path} does not have any drive report. " \
                                              "Check if it was merpped correctly.")
            elif n_reports < 3:
                LOGGER.warning(f"File {rootf.file_path} has only {n_reports} drive reports.")

            for key, leaf in drive_leaves.items():
                drive_data[key].append(drive[leaf])

        drive_data = {
            k: np.concatenate(v)
            for k, v in drive_data.items()
        }

        # convert unit as before (Better use astropy units!)
        drive_data['ra'] *= degrees_per_hour

        # get only drive reports with unique times, otherwise interpolation fails.
        _, unique_indices = np.unique(
            drive_data['mjd'],
            return_index=True
        )

        drive_data = {
            k: v[unique_indices]
            for k, v in drive_data.items()
        }

        return drive_data

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
        return self.run_numbers

    def _get_badrmspixel_mask(self, event):
        """
        Fetch the bad RMS pixel mask for a given event.

        Parameters
        ----------
        event: ctapipe.containers.ArrayEventContainer
            event container

        Returns
        -------
        badrmspixels_mask
        """

        pedestal_level = 400
        pedestal_level_variance = 4.5

        tel_id = self.telescope

        event_time = event.trigger.tel[tel_id].time.unix
        pedestal_times = event.mon.tel[tel_id].pedestal.sample_time.unix

        if np.all(pedestal_times > event_time):
            index = 0
        else:
            index = np.where(pedestal_times < event_time)[0][-1]

        badrmspixels_mask = []
        n_ped_types = len(event.mon.tel[tel_id].pedestal.charge_std)

        for i_ped_type in range(n_ped_types):

            charge_std = event.mon.tel[tel_id].pedestal.charge_std[i_ped_type][index]

            pix_step1 = np.logical_and(
                charge_std > 0,
                charge_std < 200
            )

            mean_step1 = charge_std[pix_step1].mean()

            if mean_step1 == 0:
                badrmspixels_mask.append(np.zeros(len(charge_std), np.bool))
                continue

            pix_step2 = np.logical_and(
                charge_std > 0.5 * mean_step1,
                charge_std < 1.5 * mean_step1
            )

            n_pixels = np.count_nonzero(pix_step2)
            mean_step2 = charge_std[pix_step2].mean()
            var_step2 = np.mean(charge_std[pix_step2] ** 2)

            if (n_pixels == 0) or (mean_step2 == 0):
                badrmspixels_mask.append(np.zeros(len(charge_std), np.bool))
                continue

            lolim_step1 = mean_step1 / pedestal_level
            uplim_step1 = mean_step1 * pedestal_level
            lolim_step2 = mean_step2 - pedestal_level_variance * np.sqrt(var_step2 - mean_step2 ** 2)
            uplim_step2 = mean_step2 + pedestal_level_variance * np.sqrt(var_step2 - mean_step2 ** 2)

            badrmspix_step1 = np.logical_or(
                charge_std < lolim_step1,
                charge_std > uplim_step1
            )

            badrmspix_step2 = np.logical_or(
                charge_std < lolim_step2,
                charge_std > uplim_step2
            )

            badrmspixels_mask.append(
                np.logical_or(badrmspix_step1, badrmspix_step2)
            )

        return badrmspixels_mask

    def _set_active_run(self, uproot_file):
        """
        This internal method sets the run that will be used for data loading.
        Parameters
        ----------
        uproot_file:
            The uproot file
        Returns
        -------
        run: MarsCalibratedRun
            The run to use
        """

        run = dict()
        run['read_events'] = 0
        run["run_number"] = self.obs_ids[0]

        if self.mars_datalevel == MARSDataLevel.CALIBRATED:
            run['data'] = MarsCalibratedRun(uproot_file)

        return run


    def _generator(self):
        """
        The default event generator. Return the stereo event
        generator instance.

        Returns
        -------

        """

        if self.mars_datalevel == MARSDataLevel.CALIBRATED:
            return self._event_generator()

    def _event_generator(self):
        """
        Event generator. Yields ArrayEventContainer instances,
        filled with the read event data.

        Returns
        -------
        event: ctapipe.containers.ArrayEventContainer

        """

        # Data container - is initialized once, and data is replaced after each yield
        event = ArrayEventContainer()

        event.meta['origin'] = 'MAGIC'
        event.meta['max_events'] = self.max_events

        event.index.obs_id = self.obs_ids[0]

        tel_id = self.telescope
        n_pixels = self.subarray.tel[tel_id].camera.geometry.n_pixels

        counter = 0

        # Read the input files subrun-wise
        for uproot_file in self.files_:

            if self.current_run is not None:
                del self.current_run['data']

            self.current_run = self._set_active_run(uproot_file)

            event.meta['input_file'] = uproot_file.file_path

            if self.generate_pedestals:
                event_data = self.current_run['data'].pedestal_events
                n_events = self.current_run['data'].n_pedestal_events

            else:
                event_data = self.current_run['data'].cosmic_events
                n_events = self.current_run['data'].n_cosmic_events

            if not self.is_simulation:

                monitoring_data = self.current_run['data'].monitoring_data

                # Set pedestal info:
                event.mon.tel[tel_id].pedestal.n_events = 500  # hardcoded number of pedestal events averaged over

                event.mon.tel[tel_id].pedestal.sample_time = Time(
                    monitoring_data['pedestal_sample_time'], format='unix', scale='utc'
                )

                n_samples = len(monitoring_data['pedestal_sample_time'])

                event.mon.tel[tel_id].pedestal.charge_mean = [
                    monitoring_data['pedestal_fundamental']['mean'][:,:n_pixels],
                    monitoring_data['pedestal_from_extractor']['mean'][:,:n_pixels],
                    monitoring_data['pedestal_from_extractor_rndm']['mean'][:,:n_pixels],
                ]

                event.mon.tel[tel_id].pedestal.charge_std = [
                    monitoring_data['pedestal_fundamental']['rms'][:,:n_pixels],
                    monitoring_data['pedestal_from_extractor']['rms'][:,:n_pixels],
                    monitoring_data['pedestal_from_extractor_rndm']['rms'][:,:n_pixels],
                ]

                # Set badpixel info
                event.mon.tel[tel_id].pixel_status.hardware_failing_pixels = monitoring_data['badpixel'][:n_pixels]

                # Interpolates drive information:
                drive_data = self.drive_information
                n_drive_reports = len(drive_data['mjd'])

                if self.generate_pedestals:
                    logger.warning(f'Interpolating pedestals events information from {n_drive_reports} drive reports.')
                else:
                    logger.warning(f'Interpolating cosmic events information from {n_drive_reports} drive reports.')

                first_drive_report_time = Time(drive_data['mjd'][0], scale='utc', format='mjd')
                last_drive_report_time = Time(drive_data['mjd'][-1], scale='utc', format='mjd')

                logger.warning(f'Drive reports available from {first_drive_report_time.iso} to {last_drive_report_time.iso}.')

                # Creating azimuth and zenith angles interpolators:
                drive_az_pointing_interpolator = scipy.interpolate.interp1d(
                    drive_data['mjd'], drive_data['az'], fill_value='extrapolate'
                )
                drive_zd_pointing_interpolator = scipy.interpolate.interp1d(
                    drive_data['mjd'], drive_data['zd'], fill_value='extrapolate'
                )

                # Creating RA and Dec interpolators:
                drive_ra_pointing_interpolator = scipy.interpolate.interp1d(
                    drive_data['mjd'], drive_data['ra'], fill_value='extrapolate'
                )
                drive_dec_pointing_interpolator = scipy.interpolate.interp1d(
                    drive_data['mjd'], drive_data['dec'], fill_value='extrapolate'
                )

                # Interpolating the drive pointing to the event timestamps:
                event_times = event_data['time'].mjd

                pointing_az = drive_az_pointing_interpolator(event_times)
                pointing_zd = drive_zd_pointing_interpolator(event_times)
                pointing_ra = drive_ra_pointing_interpolator(event_times)
                pointing_dec = drive_dec_pointing_interpolator(event_times)

                event_data['pointing_az'] = u.Quantity(pointing_az, u.deg)
                event_data['pointing_zd'] = u.Quantity(pointing_zd, u.deg)
                event_data['pointing_ra'] = u.Quantity(pointing_ra, u.deg)
                event_data['pointing_dec'] = u.Quantity(pointing_dec, u.deg)

            # Loop over the events
            for i_event in range(n_events):

                if self.max_events is not None:
                    if i_event > self.max_events:
                        break

                event.count = counter
                event.index.event_id = event_data['stereo_event_number'][i_event]

                event.trigger.event_type = MAGIC_TO_CTA_EVENT_TYPE.get(event_data['trigger_pattern'][i_event])
                event.trigger.tels_with_trigger = [tel_id, ]

                if not self.is_simulation:

                    event.trigger.time = event_data['time'][i_event]
                    event.trigger.tel[tel_id].time = event_data['time'][i_event]

                    badrmspixels_mask = self._get_badrmspixel_mask(event)
                    event.mon.tel[tel_id].pixel_status.pedestal_failing_pixels = badrmspixels_mask

                # Creating the telescope pointing container:
                event.pointing.array_azimuth = event_data['pointing_az'][i_event].to(u.rad)
                event.pointing.array_altitude = u.Quantity(np.pi/2, u.rad) - event_data['pointing_zd'][i_event].to(u.rad)
                event.pointing.array_ra = event_data['pointing_ra'][i_event].to(u.rad)
                event.pointing.array_dec = event_data['pointing_dec'][i_event].to(u.rad)

                event.pointing.tel[tel_id].azimuth = event_data['pointing_az'][i_event].to(u.rad)
                event.pointing.tel[tel_id].altitude = u.Quantity(np.pi/2, u.rad) - event_data['pointing_zd'][i_event].to(u.rad)

                # Adding event charge and peak positions per pixel:
                event.dl1.tel[tel_id].image = np.array(event_data['image'][i_event][:n_pixels], dtype=np.float32)
                event.dl1.tel[tel_id].peak_time = np.array(event_data['peak_time'][i_event][:n_pixels], dtype=np.float32)

                if self.is_simulation:

                    event.simulation = SimulatedEventContainer()

                    event.simulation.shower.energy = event_data['mc_energy'][i_event].to(u.TeV)
                    event.simulation.shower.shower_primary_id = 1 - event_data['mc_shower_primary_id'][i_event]
                    event.simulation.shower.h_first_int = event_data['mc_h_first_int'][i_event].to(u.m)

                    # Adding a 7 deg rotation between the orientation of corsika (x axis = magnetic north)
                    # and MARS (x axis = geographical north) frames
                    # magnetic north is 7 deg westward w.r.t. geographical north

                    mfield_dec = self.simulation_config['prod_site_B_declination']

                    event.simulation.shower.alt = u.Quantity(90, u.deg) - event_data['mc_theta'][i_event].to(u.deg)
                    event.simulation.shower.az = u.Quantity(180, u.deg) - event_data['mc_phi'][i_event].to(u.deg) + mfield_dec

                    event.simulation.shower.core_x = event_data['mc_core_x'][i_event].to(u.m) * np.cos(mfield_dec) \
                                                     + event_data['mc_core_y'][i_event].to(u.m) * np.sin(mfield_dec)

                    event.simulation.shower.core_y = event_data['mc_core_y'][i_event].to(u.m) * np.cos(mfield_dec) \
                                                     - event_data['mc_core_x'][i_event].to(u.m) * np.sin(mfield_dec)

                yield event
                counter += 1

        return


class MarsCalibratedRun:
    """
    This class implements reading of the cosmic and pedestal events,
    and the monitoring data from a MAGIC calibrated subrun file.
    """

    def __init__(self, uproot_file):
        """
        Constructor of the class. Load the input uproot file and store
        the informaiton to the constructor variables.

        Parameters
        ----------
        uproot_file: uproot.reading.ReadOnlyDirectory
            A file opened by uproot via uproot.open(file_name)
        """

        self.file_name = uproot_file.file_path

        run_info = self.parse_run_info(uproot_file)

        self.obs_id = run_info[0]
        self.tel_id = run_info[1]
        self.is_mc = run_info[2]

        # Load the input data
        calib_data = self._load_data(uproot_file)

        self.cosmic_events = calib_data['cosmic_events']
        self.pedestal_events = calib_data['pedestal_events']
        self.monitoring_data = calib_data['monitoring_data']

    @property
    def n_cosmic_events(self):
        return len(self.cosmic_events.get('trigger_pattern', []))

    @property
    def n_pedestal_events(self):
        return len(self.pedestal_events.get('trigger_pattern', []))

    @staticmethod
    def parse_run_info(uproot_file):
        """
        Parses the run info from the RunHeaders tree.

        Parameters
        ----------
        uproot_file: uproot.reading.ReadOnlyDirectory
            A file opened by uproot via uproot.open(file_name)

        Returns
        -------
        obs_id: int
            The run number of the file
        tel_id: int
            The telescope ID
        is_mc: Bool
            Flag to MC file
        """

        runheader_branches = [
            'MRawRunHeader.fRunNumber',
            'MRawRunHeader.fRunType',
            'MRawRunHeader.fTelescopeNumber',
        ]

        run_info = uproot_file['RunHeaders'].arrays(
            expressions=runheader_branches,
            library='np'
        )

        obs_id = int(run_info['MRawRunHeader.fRunNumber'][0])
        run_type = int(run_info['MRawRunHeader.fRunType'][0])
        tel_id = int(run_info['MRawRunHeader.fTelescopeNumber'][0])

        if np.isclose(run_type, mc_data_type):
            is_mc = True
        else:
            is_mc = False

        return obs_id, tel_id, is_mc

    def _load_data(self, uproot_file):
        """
        This method loads events and monitoring data from
        the input calibrated file and returns them as a dictionary.

        Parameters
        ----------
        uproot_file: uproot.reading.ReadOnlyDirectory
            A file opened by uproot via uproot.open(file_name)

        Returns
        -------
        event_data: dict
            A dictionary with the even properties,
            cosmic and pedestal events, and monitoring data
        """

        # Branches applicable for cosmic and pedestal events:
        common_branches = [
            'MCerPhotEvt.fPixels.fPhot',
            'MArrivalTime.fData',
            'MRawEvtHeader.fDAQEvtNumber',
            'MRawEvtHeader.fStereoEvtNumber',
            'MTriggerPattern.fPrescaled',
            'MTriggerPattern.fSkipEvent',
        ]

        # Branches applicable for MC events:
        mc_branches = [
            'MMcEvt.fEnergy',
            'MMcEvt.fTheta',
            'MMcEvt.fPhi',
            'MMcEvt.fPartId',
            'MMcEvt.fZFirstInteraction',
            'MMcEvt.fCoreX',
            'MMcEvt.fCoreY',
        ]

        pointing_branches = [
            'MPointingPos.fZd',
            'MPointingPos.fAz',
            'MPointingPos.fRa',
            'MPointingPos.fDec',
            'MPointingPos.fDevZd',
            'MPointingPos.fDevAz',
            'MPointingPos.fDevHa',
            'MPointingPos.fDevDec',
        ]

        # Branches applicable for real data:
        timing_branches = [
            'MTime.fMjd',
            'MTime.fTime.fMilliSec',
            'MTime.fNanoSec',
        ]

        pedestal_branches = [
            'MTimePedestals.fMjd',
            'MTimePedestals.fTime.fMilliSec',
            'MTimePedestals.fNanoSec',
            'MPedPhotFundamental.fArray.fMean',
            'MPedPhotFundamental.fArray.fRms',
            'MPedPhotFromExtractor.fArray.fMean',
            'MPedPhotFromExtractor.fArray.fRms',
            'MPedPhotFromExtractorRndm.fArray.fMean',
            'MPedPhotFromExtractorRndm.fArray.fRms',
        ]

        calib_data = {
            'cosmic_events': dict(),
            'pedestal_events': dict(),
            'monitoring_data': dict(),
        }

        events_cut = dict()

        if self.is_mc:
            # Only cosmic events because MC doesn't contain pedestal events
            events_cut['cosmic_events'] = f'(MTriggerPattern.fPrescaled == {MC_STEREO_TRIGGER_PATTERN})' \
                                          ' & (MRawEvtHeader.fStereoEvtNumber != 0)'
        else:
            events_cut['cosmic_events'] = f'(MTriggerPattern.fPrescaled == {DATA_STEREO_TRIGGER_PATTERN})'
            events_cut['pedestal_events'] = f'(MTriggerPattern.fPrescaled == {PEDESTAL_TRIGGER_PATTERN})'

        event_types = events_cut.keys()

        for event_type in event_types:

            # Reading the info common to cosmic and pedestal events:
            common_info = uproot_file['Events'].arrays(
                expressions=common_branches,
                cut=events_cut[event_type],
                library='np',
            )

            calib_data[event_type]['image'] = common_info['MCerPhotEvt.fPixels.fPhot']
            calib_data[event_type]['peak_time'] = common_info['MArrivalTime.fData']
            calib_data[event_type]['trigger_pattern'] = common_info['MTriggerPattern.fPrescaled']
            calib_data[event_type]['stereo_event_number'] = common_info['MRawEvtHeader.fStereoEvtNumber']

            if self.is_mc:

                # Reading the MC information:
                mc_info = uproot_file['Events'].arrays(
                    expressions=mc_branches,
                    cut=events_cut[event_type],
                    library='np',
                )

                # Note that the branch "MMcEvt.fPhi" is the angle between the directions
                # of the magnetic north and the angular momentum of the injected particle,
                # defined from -pi to pi (i.e., from -180 deg to 180 deg)

                calib_data[event_type]['mc_energy'] = u.Quantity(mc_info['MMcEvt.fEnergy'], u.GeV)
                calib_data[event_type]['mc_theta'] = u.Quantity(mc_info['MMcEvt.fTheta'], u.rad)
                calib_data[event_type]['mc_phi'] = u.Quantity(mc_info['MMcEvt.fPhi'], u.rad)
                calib_data[event_type]['mc_shower_primary_id'] = mc_info['MMcEvt.fPartId']
                calib_data[event_type]['mc_h_first_int'] = u.Quantity(mc_info['MMcEvt.fZFirstInteraction'], u.cm)
                calib_data[event_type]['mc_core_x'] = u.Quantity(mc_info['MMcEvt.fCoreX'], u.cm)
                calib_data[event_type]['mc_core_y'] = u.Quantity(mc_info['MMcEvt.fCoreY'], u.cm)

                # Reading the telescope pointing direction:
                pointing = uproot_file['Events'].arrays(
                    expressions=pointing_branches,
                    cut=events_cut[event_type],
                    library='np',
                )

                # ToDo: it seems the *.fDev* and *.fRa/fDec branches in the MC calibrated
                # file contain only zeros. If it is the case for all MC files, the following
                # calculation does not needed anymore and can be simply written.

                pointing_az = pointing['MPointingPos.fAz'] - pointing['MPointingPos.fDevAz']
                pointing_zd = pointing['MPointingPos.fZd'] - pointing['MPointingPos.fDevZd']

                # N.B. the positive sign here, as HA = local sidereal time - ra
                pointing_ra = (pointing['MPointingPos.fRa'] + pointing['MPointingPos.fDevHa']) * degrees_per_hour
                pointing_dec = pointing['MPointingPos.fDec'] - pointing['MPointingPos.fDevDec']

                calib_data[event_type]['pointing_az'] = u.Quantity(pointing_az, u.deg)
                calib_data[event_type]['pointing_zd'] = u.Quantity(pointing_zd, u.deg)
                calib_data[event_type]['pointing_ra'] = u.Quantity(pointing_ra, u.deg)
                calib_data[event_type]['pointing_dec'] = u.Quantity(pointing_dec, u.deg)

            else:
                # Reading the event timing information:
                timing_info = uproot_file['Events'].arrays(
                    expressions=timing_branches,
                    cut=events_cut[event_type],
                    library='np',
                )

                # In order to keep the precision of the timestamps, here the Decimal module is used.
                # At the later steps, the precise information can be extracted as follows:
                # calib_data[event_type]['time'].to_value(format='unix', subfmt='long')

                event_obs_day = Time(timing_info['MTime.fMjd'], format='mjd', scale='utc')
                event_obs_day = np.round(event_obs_day.unix)
                event_obs_day = np.array([Decimal(str(x)) for x in event_obs_day])

                event_millisec = np.round(timing_info['MTime.fTime.fMilliSec'] * msec2sec, 3)
                event_millisec = np.array([Decimal(str(x)) for x in event_millisec])

                event_nanosec = np.round(timing_info['MTime.fNanoSec'] * nsec2sec, 7)
                event_nanosec = np.array([Decimal(str(x)) for x in event_nanosec])

                event_time = event_obs_day + event_millisec + event_nanosec

                calib_data[event_type]['time'] = Time(event_time, format='unix', scale='utc')

        if not self.is_mc:

            # Reading the bad pixel info:
            as_dtype = uproot.interpretation.numerical.AsDtype(np.dtype('>i4'))
            as_jagged = uproot.interpretation.jagged.AsJagged(as_dtype)

            badpixel_info = uproot_file['RunHeaders']['MBadPixelsCam.fArray.fInfo'].array(as_jagged, library='np')[0]
            badpixel_info = badpixel_info.reshape((4, 1183), order='F')

            # now we have 4 axes:
            # 0st axis: empty (?)
            # 1st axis: Unsuitable pixels
            # 2nd axis: Uncalibrated pixels (says why pixel is unsuitable)
            # 3rd axis: Bad hardware pixels (says why pixel is unsuitable)
            # Each axis cointains a 32bit integer encoding more information about the specific problem
            # See MARS software, MBADPixelsPix.h for more information
            # Here we take the first axis:

            unsuitable_pix_bitinfo = badpixel_info[1]
            unsuitable_pix = np.zeros(len(unsuitable_pix_bitinfo), dtype=bool)

            for i_pix in range(len(unsuitable_pix_bitinfo)):
                unsuitable_pix[i_pix] = int('{:08b}'.format(unsuitable_pix_bitinfo[i_pix])[-2])

            calib_data['monitoring_data']['badpixel'] = unsuitable_pix

            # Try to read the Pedestals tree (soft fail if not present)
            try:
                pedestal_info = uproot_file['Pedestals'].arrays(
                    expressions=pedestal_branches,
                    library='np',
                )

                pedestal_obs_day = Time(pedestal_info['MTimePedestals.fMjd'], format='mjd', scale='utc')
                pedestal_obs_day = np.round(pedestal_obs_day.unix)
                pedestal_obs_day = np.array([Decimal(str(x)) for x in pedestal_obs_day])

                pedestal_millisec = np.round(pedestal_info['MTimePedestals.fTime.fMilliSec'] * msec2sec, 3)
                pedestal_millisec = np.array([Decimal(str(x)) for x in pedestal_millisec])

                pedestal_nanosec = np.round(pedestal_info['MTimePedestals.fNanoSec'] * nsec2sec, 7)
                pedestal_nanosec = np.array([Decimal(str(x)) for x in pedestal_nanosec])

                pedestal_sample_time = pedestal_obs_day + pedestal_millisec + pedestal_nanosec

                calib_data['monitoring_data']['pedestal_sample_time'] = Time(
                    pedestal_sample_time, format='unix', scale='utc'
                )

                n_samples = len(pedestal_sample_time)

                calib_data['monitoring_data']['pedestal_fundamental'] = {
                    'mean': np.reshape(pedestal_info[f'MPedPhotFundamental.fArray.fMean'].tolist(), (n_samples, 1183)),
                    'rms': np.reshape(pedestal_info[f'MPedPhotFundamental.fArray.fRms'].tolist(), (n_samples, 1183)),
                }
                calib_data['monitoring_data']['pedestal_from_extractor'] = {
                    'mean': np.reshape(pedestal_info[f'MPedPhotFromExtractor.fArray.fMean'].tolist(), (n_samples, 1183)),
                    'rms': np.reshape(pedestal_info[f'MPedPhotFromExtractor.fArray.fRms'].tolist(), (n_samples, 1183)),
                }
                calib_data['monitoring_data']['pedestal_from_extractor_rndm'] = {
                    'mean': np.reshape(pedestal_info[f'MPedPhotFromExtractorRndm.fArray.fMean'].tolist(), (n_samples, 1183)),
                    'rms': np.reshape(pedestal_info[f'MPedPhotFromExtractorRndm.fArray.fRms'].tolist(), (n_samples, 1183)),
                }

            except KeyError:
                logger.warning('Pedestals tree is not present in the input file. Cleaning algorithm may fail.')

            # Check for bit flips in the stereo event ID:
            max_total_jumps = 100

            stereo_event_number = calib_data['cosmic_events']['stereo_event_number'].astype(int)
            number_difference = np.diff(stereo_event_number)

            indices_flip = np.where(number_difference < 0)[0]
            n_flips = len(indices_flip)

            if n_flips > 0:

                logger.warning(f'Warning: detected {n_flips} bitflips in file {self.file_name}')
                total_jumped_number = 0

                for i in indices_flip:

                    jumped_number = stereo_event_number[index] - stereo_event_number[i+1]
                    total_jumped_number += jumped_number

                    logger.warning(f'Jump of L3 number backward from {stereo_event_number[i]} to ' \
                                   f'{stereo_event_number[i+1]}; total jumped events so far: {total_jumped_number}')

                if total_jumped_number > max_total_jumps:
                    logger.warning(f'More than {max_total_jumps} in stereo trigger number; ' \
                                   f'you may have to match events by timestamp at a later stage.')

        return calib_data
