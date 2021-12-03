"""
# Event source for MAGIC calibrated data files.
# Requires uproot package (https://github.com/scikit-hep/uproot).
"""

import logging

import glob
import re
import os.path
from pathlib import Path
from enum import Enum, auto

import numpy as np

import scipy
import scipy.interpolate

from astropy.coordinates import Angle
from astropy import units as u
from astropy.time import Time

from ctapipe.io.eventsource import EventSource
from ctapipe.io.datalevels import DataLevel

from ctapipe.core import Container
from ctapipe.core import Field

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

__all__ = ['MAGICEventSource']

LOGGER = logging.getLogger(__name__)


# MAGIC telescope positions in m wrt. to the center of CTA simulations
#MAGIC_TEL_POSITIONS = {
#    1: [-27.24, -146.66, 50.00] * u.m,
#    2: [-96.44, -96.77, 51.00] * u.m
#}

# MAGIC telescope positions in m wrt. to the center of MAGIC simulations, from reflector input card
MAGIC_TEL_POSITIONS = {
    1: [31.80, -28.10, 0.00] * u.m,
    2: [-31.80, 28.10, 0.00] * u.m
}

# MAGIC telescope description
OPTICS = OpticsDescription.from_name('MAGIC')
MAGICCAM = CameraDescription.from_name("MAGICCam")
pulse_shape_lo_gain = np.array([])
pulse_shape_hi_gain = np.array([])
pulse_shape = np.vstack((pulse_shape_lo_gain, pulse_shape_lo_gain))
MAGICCAM.readout = CameraReadout(
    camera_name='MAGICCam',
    sampling_rate=u.Quantity(1.64, u.GHz),
    reference_pulse_shape=pulse_shape,
    reference_pulse_sample_width=u.Quantity(0.5, u.ns)
)
GEOM = MAGICCAM.geometry
MAGIC_TEL_DESCRIPTION = TelescopeDescription(
    name='MAGIC', tel_type='MAGIC', optics=OPTICS, camera=MAGICCAM)
MAGIC_TEL_DESCRIPTIONS = {1: MAGIC_TEL_DESCRIPTION, 2: MAGIC_TEL_DESCRIPTION}

# trigger patterns:
MC_TRIGGER_PATTERN = 1
PEDESTAL_TRIGGER_PATTERN = 8
DATA_TRIGGER_PATTERN = 128

class MARSDataLevel(Enum):
    """Enum of the different MARS Data Levels"""

    CALIBRATED = auto()  # Calibrated images in charge and time (no waveforms)
    STAR       = auto()  # Cleaned images, with Hillas parametrization
    SUPERSTAR  = auto()  # Stereo parameters reconstructed
    MELIBEA    = auto()  # Reconstruction of hadronness, event direction and energy

class L3JumpError(Exception):
    """
    Exception raised when L3 trigger number jumps backward.
    """

    def __init__(self, message):
        self.message = message

class MissingDriveReportError(Exception):
    """
    Exception raised when a subrun does not have drive reports.
    """

    def __init__(self, message):
        self.message = message

class MAGICEventSource(EventSource):
    """
    EventSource for MAGIC calibrated data.

    This class operates with the MAGIC data run-wise. This means that the files
    corresponding to the same data run are loaded and processed together.
    """
    _count = 0

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
        run_info = self.get_run_info_from_name(str(input_url))
        run_numbers = run_info[0]
        is_mc_runs = run_info[1]
        telescope = run_info[2]

        self.run_numbers = run_numbers
        self.is_mc = is_mc_runs
        self.telescope = telescope

        # Retrieving the data level (so far HARDCODED Sorcerer)
        self.datalevel = DataLevel.DL1_IMAGES
        self.mars_datalevel = run_info[3]

        if self.is_mc:
            self.simulation_config = self.parse_simulation_header()

        # # Setting up the current run with the first run present in the data
        # self.current_run = self._set_active_run(run_number=0)
        self.current_run = None

        self._subarray_info = SubarrayDescription(
            'MAGIC', MAGIC_TEL_POSITIONS, MAGIC_TEL_DESCRIPTIONS)

    @staticmethod
    def is_compatible(file_mask):
        """
        This method checks if the specified file mask corresponds
        to MAGIC data files. The result will be True only if all
        the files are of ROOT format and contain an 'Events' tree.

        Parameters
        ----------
        file_mask: str
            A file mask to check

        Returns
        -------
        bool:
            True if the masked files are MAGIC data runs, False otherwise.

        """

        is_magic_root_file = True

        file_list = glob.glob(str(file_mask))

        for file_path in file_list:
            try:
                import uproot

                try:
                    with uproot.open(file_path) as input_data:
                        if 'Events' not in input_data:
                            is_magic_root_file = False
                except ValueError:
                    # uproot raises ValueError if the file is not a ROOT file
                    is_magic_root_file = False

            except ImportError:
                if re.match(r'.+_m\d_.+root', file_path.lower()) is None:
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
        mask_mc_alt = r".*_M(\d)_\d_(\d+)_.*"
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

    def parse_simulation_header(self):

        try:
            import uproot
            with uproot.open(self.input_url) as input_file:
                run_header_tree = input_file['RunHeaders']
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
                    shower_reuse=1, #not written in the magic root file, but since the sim_events already include shower reuse we artificially set it to 1 (actually every shower reused 5 times for std MAGIC MC)
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

        except ImportError:
            print("Error while importing uproot.")

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
            run['data'] = MarsCalibratedRun(run_file_mask=str(self.input_url))
        elif self.mars_datalevel == MARSDataLevel.MELIBEA:
            run['data'] = MarsMelibeaRun(run_file_mask=str(self.input_url))

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
            return self._mono_event_generator(telescope=f"M{self.telescope}")
        elif self.mars_datalevel == MARSDataLevel.MELIBEA:
            return self._melibea_event_generator()

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
        tels_with_data = {1, 2}

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

                    time_tmp = Time(monitoring_data['M{:d}'.format(
                        tel_i + 1)]['PedestalMJD'], scale='utc', format='mjd')
                    pedestal_info.sample_time = Time(
                        time_tmp, format='unix', scale='utc', precision=9)
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

                    t_range = Time(monitoring_data['M{:d}'.format(
                        tel_i + 1)]['badpixelinfoMJDrange'], scale='utc', format='mjd')

                    badpixel_info.hardware_failing_pixels = monitoring_data['M{:d}'.format(
                        tel_i + 1)]['badpixelinfo']
                    badpixel_info.sample_time_range = t_range

                    monitoring_camera.pedestal = pedestal_info
                    monitoring_camera.pixel_status = badpixel_info

                    data.mon.tels_with_data = {1, 2}
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
                event_data = self.current_run['data'].get_stereo_event_data(
                    event_i)

                data.meta = event_data['mars_meta']

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
                pointing.array_dec = np.deg2rad(90 - event_data['m1_pointing_dec']) * u.rad
                data.pointing = pointing

                if not self.is_mc:
                    # Adding the event arrival time
                    time_tmp = Time(
                        event_data['mjd'], scale='utc', format='mjd')
                    data.trigger.time = Time(
                        time_tmp, format='unix', scale='utc', precision=9)
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

        # Data container - is initialized once, and data is replaced within it after each yield
        data = ArrayEventContainer()

        # Telescopes with data:
        tels_in_file = ["M1", "M2"]

        if telescope not in tels_in_file:
            raise ValueError("Specified telescope {:s} is not in the allowed list {}".format(
                telescope, tels_in_file))

        tel_i = tels_in_file.index(telescope)
        tels_with_data = {tel_i + 1, }

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

            time_tmp = Time(monitoring_data['M{:d}'.format(
                tel_i + 1)]['PedestalMJD'], scale='utc', format='mjd')
            pedestal_info.sample_time = Time(
                time_tmp, format='unix', scale='utc', precision=9)
            pedestal_info.n_events = 500 # hardcoded number of pedestal events averaged over
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

            t_range = Time(monitoring_data['M{:d}'.format(
                tel_i + 1)]['badpixelinfoMJDrange'], scale='utc', format='mjd')

            badpixel_info.hardware_failing_pixels = monitoring_data['M{:d}'.format(
                tel_i + 1)]['badpixelinfo']
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
            event_data = self.current_run['data'].get_mono_event_data(
                event_i, telescope=telescope)

            data.meta = event_data['mars_meta']
            data.meta["max_events"] = self.max_events

            data.trigger.event_type = self.current_run['data'].event_data[telescope]['trigger_pattern'][event_order_number]
            data.trigger.tels_with_trigger = tels_with_data
            if self.allowed_tels:
                data.trigger.tels_with_trigger = np.intersect1d(
                    data.trigger.tels_with_trigger,
                    self.subarray.tel_ids,
                    assume_unique=True,)
            if not self.is_mc:
                # Adding the event arrival time
                time_tmp = Time(
                    event_data['mjd'], scale='utc', format='mjd')
                data.trigger.tel[tel_i + 1] = TelescopeTriggerContainer(time=Time(
                    time_tmp, format='unix', scale='utc', precision=9))

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
                azimuth = np.deg2rad(
                event_data['pointing_az']) * u.rad,
                altitude = np.deg2rad(
                90 - event_data['pointing_zd']) * u.rad,)

            pointing.tel[tel_i + 1] = pointing_tel

            pointing.array_azimuth  = np.deg2rad(event_data['pointing_az']) * u.rad
            pointing.array_altitude = np.deg2rad(90 - event_data['pointing_zd']) * u.rad
            pointing.array_ra       = np.deg2rad(event_data['pointing_ra']) * u.rad
            pointing.array_dec      = np.deg2rad(90 - event_data['pointing_dec']) * u.rad

            data.pointing = pointing

            # Adding event charge and peak positions per pixel
            data.dl1.tel[tel_i + 1].image = event_data['image']
            data.dl1.tel[tel_i + 1].peak_time = event_data['pulse_time']

            if self.is_mc:
                rot_corsika = 7 *u.deg
                # check meaning of 7deg transformation (I.Vovk)
                # adding a 7deg rotation between the orientation of corsika (x axis = magnetic north) and MARS (x axis = geographical north) frames
                # magnetic north is 7 deg westward w.r.t. geographical north
                data.simulation = SimulatedEventContainer()
                data.simulation.shower = SimulatedShowerContainer(
                    energy = u.Quantity(event_data['true_energy'], u.GeV),
                    alt = Angle((np.pi/2 - event_data['true_zd']), u.rad),
                    az = Angle(-1 * (event_data['true_az'] - np.deg2rad(180 - rot_corsika.value)), u.rad),
                    shower_primary_id = 1 - event_data['true_shower_primary_id'],
                    h_first_int = u.Quantity(event_data['true_h_first_int'], u.cm),
                    core_x = u.Quantity((event_data['true_core_x']*np.cos(rot_corsika) - event_data['true_core_y']*np.sin(rot_corsika)).value, u.cm),
                    core_y = u.Quantity((event_data['true_core_x']*np.sin(rot_corsika) + event_data['true_core_y']*np.cos(rot_corsika)).value, u.cm),
                )

            yield data
            counter += 1

        return

    def _melibea_event_generator(self):
        """
        Melibea event generator. Yields ArrayEventContainer instances, filled
        with the read event data.
        """

        counter = 0

        # Data container - is initialized once, and data is replaced within it after each yield
        data = ArrayEventContainer()

        # Telescopes with data:
        tels_in_file = ["M1", "M2"]

        tels_with_data = {1, 2}

        # Removing the previously read data run from memory
        if self.current_run is not None:
            if 'data' in self.current_run:
                del self.current_run['data']

        # Setting the new active run
        self.current_run = self._set_active_run(self.run_numbers)
        event_data = self.current_run.event_data

        # Loop over the events
        for event in range(n_events):
            # Event and run ids
            event_order_number = self.current_run['data'].mono_ids[telescope][event_i]
            event_id = self.current_run['data'].event_data[telescope]['stereo_event_number'][event_order_number]
            obs_id = self.current_run['number']

            # Reading event data
            event_data = self.current_run['data'].get_mono_event_data(
                event_i, telescope=telescope)

            data.meta = event_data['mars_meta']
            data.meta["max_events"] = self.max_events

            data.trigger.event_type = self.current_run['data'].event_data[telescope]['trigger_pattern'][event_order_number]
            data.trigger.tels_with_trigger = tels_with_data
            if self.allowed_tels:
                data.trigger.tels_with_trigger = np.intersect1d(
                    data.trigger.tels_with_trigger,
                    self.subarray.tel_ids,
                    assume_unique=True,)
            if not self.is_mc:
                # Adding the event arrival time
                time_tmp = Time(
                    event_data['mjd'], scale='utc', format='mjd')
                data.trigger.tel[tel_i + 1] = TelescopeTriggerContainer(time=Time(
                    time_tmp, format='unix', scale='utc', precision=9))

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
                azimuth = np.deg2rad(
                event_data['pointing_az']) * u.rad,
                altitude = np.deg2rad(
                90 - event_data['pointing_zd']) * u.rad,)

            pointing.tel[tel_i + 1] = pointing_tel

            pointing.array_azimuth  = np.deg2rad(event_data['pointing_az']) * u.rad
            pointing.array_altitude = np.deg2rad(90 - event_data['pointing_zd']) * u.rad
            pointing.array_ra       = np.deg2rad(event_data['pointing_ra']) * u.rad
            pointing.array_dec      = np.deg2rad(90 - event_data['pointing_dec']) * u.rad

            data.pointing = pointing

            # Adding event charge and peak positions per pixel
            data.dl1.tel[tel_i + 1].image = event_data['image']
            data.dl1.tel[tel_i + 1].peak_time = event_data['pulse_time']

            if self.is_mc:
                rot_corsika = 7 *u.deg
                # check meaning of 7deg transformation (I.Vovk)
                # adding a 7deg rotation between the orientation of corsika (x axis = magnetic north) and MARS (x axis = geographical north) frames
                # magnetic north is 7 deg westward w.r.t. geographical north
                data.simulation.shower = SimulatedShowerContainer(
                    energy = u.Quantity(event_data['true_energy'], u.GeV),
                    alt = Angle((np.pi/2 - event_data['true_zd']), u.rad),
                    az = Angle(-1 * (event_data['true_az'] - np.deg2rad(180 - rot_corsika.value)), u.rad),
                    shower_primary_id = 1 - event_data['true_shower_primary_id'],
                    h_first_int = u.Quantity(event_data['true_h_first_int'], u.cm),
                    core_x = u.Quantity((event_data['true_core_x']*np.cos(rot_corsika) - event_data['true_core_y']*np.sin(rot_corsika)), u.cm),
                    core_y = u.Quantity((event_data['true_core_x']*np.sin(rot_corsika) + event_data['true_core_y']*np.cos(rot_corsika)), u.cm),
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

        # Data container - is initialized once, and data is replaced within it after each yield
        data = EventAndMonDataContainer()

        # Telescopes with data:
        tels_in_file = ["M1", "M2"]

        if telescope not in tels_in_file:
            raise ValueError("Specified telescope {:s} is not in the allowed list {}".format(
                telescope, tels_in_file))

        tel_i = tels_in_file.index(telescope)
        tels_with_data = {tel_i + 1, }

        # Loop over the available data runs
        for run_number in self.run_numbers:

            # Removing the previously read data run from memory
            if self.current_run is not None:
                if 'data' in self.current_run:
                    del self.current_run['data']

            # Setting the new active run
            self.current_run = self._set_active_run(run_number)

            monitoring_data = self.current_run['data'].monitoring_data

            monitoring_camera = MonitoringCameraContainer()
            pedestal_info = PedestalContainer()
            badpixel_info = PixelStatusContainer()

            time_tmp = Time(monitoring_data['M{:d}'.format(
                tel_i + 1)]['PedestalMJD'], scale='utc', format='mjd')
            pedestal_info.sample_time = Time(
                time_tmp, format='unix', scale='utc', precision=9)
            pedestal_info.n_events = 500 # hardcoded number of pedestal events averaged over
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

            t_range = Time(monitoring_data['M{:d}'.format(
                tel_i + 1)]['badpixelinfoMJDrange'], scale='utc', format='mjd')

            badpixel_info.hardware_failing_pixels = monitoring_data['M{:d}'.format(
                tel_i + 1)]['badpixelinfo']
            badpixel_info.sample_time_range = t_range

            monitoring_camera.pedestal = pedestal_info
            monitoring_camera.pixel_status = badpixel_info

            data.mon.tels_with_data = tels_with_data
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

                data.meta = event_data['mars_meta']

                # Event counter
                data.count = counter
                data.index.obs_id = obs_id
                data.index.event_id = event_id

                # Setting up the R0 container
                data.r0.tel.clear()
                data.r0.tel[tel_i + 1].trigger_type = self.current_run['data'].event_data[telescope]['trigger_pattern'][event_order_number]

                # Setting up the R1 container
                data.r1.tel.clear()
                data.r1.tel[tel_i + 1].trigger_type = self.current_run['data'].event_data[telescope]['trigger_pattern'][event_order_number]

                # Setting up the DL0 container
                data.dl0.tel.clear()
                data.dl0.tel[tel_i + 1].trigger_type = self.current_run['data'].event_data[telescope]['trigger_pattern'][event_order_number]

                # Creating the telescope pointing container
                pointing = TelescopePointingContainer()
                pointing.azimuth = np.deg2rad(
                    event_data['pointing_az']) * u.rad
                pointing.altitude = np.deg2rad(
                    90 - event_data['pointing_zd']) * u.rad
                #pointing.ra = np.deg2rad(event_data['pointing_ra']) * u.rad
                #pointing.dec = np.deg2rad(event_data['pointing_dec']) * u.rad

                # Adding the pointing container to the event data
                data.pointing.tel[tel_i + 1] = pointing

                # Adding event charge and peak positions per pixel
                data.dl1.tel[tel_i + 1].image = event_data['image']
                data.dl1.tel[tel_i + 1].peak_time = event_data['pulse_time']

                # Adding the event arrival time
                time_tmp = Time(event_data['mjd'], scale='utc', format='mjd')
                data.trigger.time = Time(
                    time_tmp, format='unix', scale='utc', precision=9)

                # Setting the telescopes with data
                data.r0.tels_with_data = tels_with_data
                data.r1.tels_with_data = tels_with_data
                data.dl0.tels_with_data = tels_with_data
                data.trigger.tels_with_trigger = tels_with_data

                yield data
                counter += 1

        return


class MarsCalibratedRun:
    """
    This class implements reading of the event data from a single MAGIC calibrated run.
    """

    def __init__(self, run_file_mask, filter_list=None):
        """
        Constructor of the class. Defines the run to use and the camera pixel arrangement.

        Parameters
        ----------
        run_file_mask: str
            A path mask for files belonging to the run. Must correspond to a single run
            or an exception will be raised. Must correspond to calibrated ("sorcerer"-level)
            data.
        filter_list: list, optional
            A list of files, to which the run_file_mask should be applied. If None, all the
            files satisfying run_file_mask will be used. Defaults to None.
        """

        self.n_camera_pixels = GEOM.n_pixels

        self.run_file_mask = run_file_mask

        # Preparing the lists of M1/2 data files
        file_list = glob.glob(run_file_mask)

        # Filtering out extra files if necessary
        if filter_list is not None:
            file_list = list(set(file_list) & set(filter_list))

        self.m1_file_list = list(
            filter(lambda name: '_M1_' in name, file_list))
        self.m2_file_list = list(
            filter(lambda name: '_M2_' in name, file_list))
        self.m1_file_list.sort()
        self.m2_file_list.sort()

        # Retrieving the list of run numbers corresponding to the data files
        run_info = list(
            map(MAGICEventSource.get_run_info_from_name, file_list))
        run_numbers = [i[0] for i in run_info]
        is_mc_runs = [i[1] for i in run_info]

        run_numbers = np.unique(run_numbers)
        is_mc_runs = np.unique(is_mc_runs)
        # Checking if run type (data/MC) is consistent:
        if len(is_mc_runs) > 1:
            raise ValueError(
                "Run type is not consistently data or MC: {}".format(is_mc_runs))

        self.is_mc = is_mc_runs[0]

        # Checking if a single run is going to be read
        if len(run_numbers) > 1:
            raise ValueError(
                "Run mask corresponds to more than one run: {}".format(run_numbers))

        # Reading the data
        m1_data = self.load_events(
            self.m1_file_list, self.is_mc, self.n_camera_pixels)
        m2_data = self.load_events(
            self.m2_file_list, self.is_mc, self.n_camera_pixels)

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
        return len(self.event_data['M1']['MJD'])

    @property
    def n_events_m2(self):
        return len(self.event_data['M2']['MJD'])

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
    def load_events(file_list, is_mc, n_camera_pixels):
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

        try:
            import uproot
        except ImportError:
            msg = "The `uproot` python module is required to access the MAGIC data"
            raise ImportError(msg)

        event_data = dict()

        event_data['charge'] = []
        event_data['arrival_time'] = []
        event_data['trigger_pattern'] = np.array([], dtype=np.int32)
        event_data['stereo_event_number'] = np.array([], dtype=np.int32)
        event_data['pointing_zd'] = np.array([])
        event_data['pointing_az'] = np.array([])
        event_data['pointing_ra'] = np.array([])
        event_data['pointing_dec'] = np.array([])
        event_data['MJD'] = np.array([])
        event_data['mars_meta'] = []

        # monitoring information (updated from time to time)
        monitoring_data = dict()

        monitoring_data['badpixelinfo'] = []
        monitoring_data['badpixelinfoMJDrange'] = []
        monitoring_data['PedestalMJD'] = np.array([])
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
        if len(file_list) == 0:
            return event_data, monitoring_data

        drive_data = dict()
        drive_data['mjd'] = np.array([])
        drive_data['zd']  = np.array([])
        drive_data['az']  = np.array([])
        drive_data['ra']  = np.array([])
        drive_data['dec'] = np.array([])

        degrees_per_hour = 15.0
        seconds_per_day = 86400.0
        seconds_per_hour = 3600.

        evt_common_list = [
            'MCerPhotEvt.fPixels.fPhot',
            'MArrivalTime.fData',
            'MTriggerPattern.fPrescaled',
            'MRawEvtHeader.fStereoEvtNumber',
            'MRawEvtHeader.fDAQEvtNumber',
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

        # Metadata, currently not strictly required
        metainfo_array_list = [
            'MRawRunHeader.fRunNumber',
            'MRawRunHeader.fRunType',
            'MRawRunHeader.fSubRunIndex',
            'MRawRunHeader.fSourceRA',
            'MRawRunHeader.fSourceDEC',
            'MRawRunHeader.fTelescopeNumber']

        for file_name in file_list:

            input_file = uproot.open(file_name)

            events = input_file['Events'].arrays(evt_common_list, library="np")

            # Reading the info common to MC and real data
            charge = events['MCerPhotEvt.fPixels.fPhot']
            arrival_time = events['MArrivalTime.fData']
            trigger_pattern = events['MTriggerPattern.fPrescaled']
            stereo_event_number = events['MRawEvtHeader.fStereoEvtNumber']

            # Reading run-wise meta information (same for all events in subrun):
            mars_meta = dict()

            mars_meta['is_simulation'] = is_mc

            # Reading event timing information:
            if not is_mc:
                event_times = input_file['Events'].arrays(time_array_list, library="np")
                # Computing the event arrival time

                event_mjd = event_times['MTime.fMjd']
                event_millisec = event_times['MTime.fTime.fMilliSec']
                event_nanosec = event_times['MTime.fNanoSec']

                event_mjd = event_mjd + \
                    (event_millisec / 1e3 + event_nanosec / 1e9) / seconds_per_day
                event_data['MJD'] = np.concatenate(
                    (event_data['MJD'], event_mjd))

            # try to read RunHeaders tree (soft fail if not present, to pass current tests)
            try:
                meta_info = input_file['RunHeaders'].arrays(
                    metainfo_array_list, library="np")

                mars_meta['origin'] = "MAGIC"
                mars_meta['input_url'] = file_name

                mars_meta['number'] = int(
                    meta_info['MRawRunHeader.fRunNumber'][0])
                mars_meta['number_subrun'] = int(
                    meta_info['MRawRunHeader.fSubRunIndex'][0])
                mars_meta['source_ra'] = meta_info['MRawRunHeader.fSourceRA'][0] / \
                    seconds_per_hour * degrees_per_hour * u.deg
                mars_meta['source_dec'] = meta_info['MRawRunHeader.fSourceDEC'][0] / \
                    seconds_per_hour * u.deg

                is_mc_check = int(meta_info['MRawRunHeader.fRunType'][0])
                if is_mc_check == 0:
                    is_mc_check = False
                elif is_mc_check == 256:
                    is_mc_check = True
                else:
                    msg = "Run type (Data or MC) of MAGIC data file not recognised."
                    LOGGER.error(msg)
                    raise ValueError(msg)
                if is_mc_check != is_mc:
                    msg = "Inconsistent run type (data or MC) between file name and runheader content."
                    LOGGER.error(msg)
                    raise ValueError(msg)

                # Reading the info only contained in real data
                if not is_mc:
                    badpixelinfo = input_file['RunHeaders']['MBadPixelsCam.fArray.fInfo'].array(
                        uproot.interpretation.jagged.AsJagged(uproot.interpretation.numerical.AsDtype(np.dtype('>i4'))), library="np")[0].reshape((4, 1183), order='F')
                    # now we have 4 axes:
                    # 0st axis: empty (?)
                    # 1st axis: Unsuitable pixels
                    # 2nd axis: Uncalibrated pixels (says why pixel is unsuitable)
                    # 3rd axis: Bad hardware pixels (says why pixel is unsuitable)
                    # Each axis cointains a 32bit integer encoding more information about the specific problem, see MARS software, MBADPixelsPix.h
                    # take first axis
                    unsuitable_pix_bitinfo = badpixelinfo[1][:n_camera_pixels]
                    # extract unsuitable bit:
                    unsuitable_pix = np.zeros(n_camera_pixels, dtype=np.bool)
                    for i in range(n_camera_pixels):
                        unsuitable_pix[i] = int('\t{0:08b}'.format(
                            unsuitable_pix_bitinfo[i] & 0xff)[-2])
                    monitoring_data['badpixelinfo'].append(unsuitable_pix)
                    # save time interval of badpixel info:
                    monitoring_data['badpixelinfoMJDrange'].append(
                        [event_mjd[0], event_mjd[-1]])

            except KeyError:
                LOGGER.warning(
                    "RunHeaders tree not present in file. Cannot read meta information - will assume it is a real data run.")
                is_mc = False

            # try to read Pedestals tree (soft fail if not present)
            if not is_mc:
                try:
                    pedestal_info = input_file['Pedestals'].arrays(
                        pedestal_array_list, library="np")

                    pedestal_mjd = pedestal_info['MTimePedestals.fMjd']
                    pedestal_millisec = pedestal_info['MTimePedestals.fTime.fMilliSec']
                    pedestal_nanosec = pedestal_info['MTimePedestals.fNanoSec']
                    n_pedestals = len(pedestal_mjd)
                    pedestal_mjd = pedestal_mjd + \
                        (pedestal_millisec / 1e3 +
                         pedestal_nanosec / 1e9) / seconds_per_day
                    monitoring_data['PedestalMJD'] = np.concatenate(
                        (monitoring_data['PedestalMJD'], pedestal_mjd))
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
            else:
                # Getting the telescope drive info
                drive = input_file['Drive'].arrays(drive_array_list, library="np")

                drive_mjd = drive['MReportDrive.fMjd']
                drive_zd = drive['MReportDrive.fCurrentZd']
                drive_az = drive['MReportDrive.fCurrentAz']
                drive_ra = drive['MReportDrive.fRa'] * degrees_per_hour
                drive_dec = drive['MReportDrive.fDec']

                drive_data['mjd'] = np.concatenate((drive_data['mjd'],drive_mjd))
                drive_data['zd']  = np.concatenate((drive_data['zd'],drive_zd))
                drive_data['az']  = np.concatenate((drive_data['az'],drive_az))
                drive_data['ra']  = np.concatenate((drive_data['ra'],drive_ra))
                drive_data['dec'] = np.concatenate((drive_data['dec'],drive_dec))

                if len(drive_mjd) < 3:
                    LOGGER.warning(f"File {file_name} has only {len(drive_mjd)} drive reports.")
                    if len(drive_mjd) == 0:
                        raise MissingDriveReportError(f"File {file_name} does not have any drive report. Check if it was merpped correctly.")

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
                    len(dx_flip_ids_before), file_name))
                total_jumped_events = 0
                for i in dx_flip_ids_before:
                    trigger_pattern[i] = -1
                    trigger_pattern[i+1] = -1
                    if not is_mc:
                        jumped_events = int(stereo_event_number[i]) - int(stereo_event_number[i+1])
                        total_jumped_events += jumped_events
                        LOGGER.warning(f"Jump of L3 number backward from {stereo_event_number[i]} to {stereo_event_number[i+1]}; "
                            f"total jumped events so far: {total_jumped_events}")
                        if total_jumped_events > max_total_jumps:
                            raise L3JumpError(f"Jumps backward in L3 trigger number by {total_jumped_events} in total. You might consider matching events by time instead.")

            event_data['charge'].append(charge)
            event_data['arrival_time'].append(arrival_time)
            event_data['mars_meta'].append(mars_meta)
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
            monitoring_data['badpixelinfo'] = np.array(
                monitoring_data['badpixelinfo'])
            monitoring_data['badpixelinfoMJDrange'] = np.array(
                monitoring_data['badpixelinfoMJDrange'])
            # sort monitoring data:
            order = np.argsort(monitoring_data['PedestalMJD'])
            monitoring_data['PedestalMJD'] = monitoring_data['PedestalMJD'][order]

            for quantity in ['Mean', 'Rms']:
                monitoring_data['PedestalFundamental'][quantity] = np.array(
                    monitoring_data['PedestalFundamental'][quantity])
                monitoring_data['PedestalFromExtractor'][quantity] = np.array(
                    monitoring_data['PedestalFromExtractor'][quantity])
                monitoring_data['PedestalFromExtractorRndm'][quantity] = np.array(
                    monitoring_data['PedestalFromExtractorRndm'][quantity])

            # get only drive reports with unique times, otherwise interpolation fails.
            drive_mjd_unique, unique_indices = np.unique(drive_data['mjd'], return_index=True)
            drive_zd_unique  = drive_data['zd'][unique_indices]
            drive_az_unique  = drive_data['az'][unique_indices]
            drive_ra_unique  = drive_data['ra'][unique_indices]
            drive_dec_unique = drive_data['dec'][unique_indices]

            first_drive_report_time = Time(drive_mjd_unique[0], scale='utc', format='mjd')
            last_drive_report_time  = Time(drive_mjd_unique[-1], scale='utc', format='mjd')

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
            event_data['pointing_zd'] = drive_zd_pointing_interpolator(event_data['MJD'])
            event_data['pointing_az'] = drive_az_pointing_interpolator(event_data['MJD'])
            event_data['pointing_ra'] = drive_ra_pointing_interpolator(event_data['MJD'])
            event_data['pointing_dec'] = drive_dec_pointing_interpolator(event_data['MJD'])

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
            stereo_m1_data = self.event_data['M1']['stereo_event_number'][np.where(self.event_data['M1']['trigger_pattern'] == DATA_TRIGGER_PATTERN)]
            stereo_m2_data = self.event_data['M2']['stereo_event_number'][np.where(self.event_data['M2']['trigger_pattern'] == DATA_TRIGGER_PATTERN)]

            # find common values between M1 and M2 stereo events, see https://numpy.org/doc/stable/reference/generated/numpy.intersect1d.html
            stereo_numbers = np.intersect1d(stereo_m1_data, stereo_m2_data)

            # find indices of the stereo event numbers in original stereo event numbers arrays, see
            # https://stackoverflow.com/questions/12122639/find-indices-of-a-list-of-values-in-a-numpy-array
            m1_ids = np.searchsorted(self.event_data['M1']['stereo_event_number'], stereo_numbers)
            m2_ids = np.searchsorted(self.event_data['M2']['stereo_event_number'], stereo_numbers)

            # make list of tuples, see https://stackoverflow.com/questions/2407398/how-to-merge-lists-into-a-list-of-tuples
            stereo_ids = list(zip(m1_ids, m2_ids))
        else:
            stereo_m1_data = self.event_data['M1']['stereo_event_number'][np.where(self.event_data['M1']['trigger_pattern'] == MC_TRIGGER_PATTERN)]
            stereo_m2_data = self.event_data['M2']['stereo_event_number'][np.where(self.event_data['M2']['trigger_pattern'] == MC_TRIGGER_PATTERN)]
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
                m1_data = self.event_data['M1']['stereo_event_number'][np.where(self.event_data['M1']['trigger_pattern'] == DATA_TRIGGER_PATTERN)]
                m2_data = self.event_data['M2']['stereo_event_number'][np.where(self.event_data['M2']['trigger_pattern'] == DATA_TRIGGER_PATTERN)]

                m1_ids_data = np.where(self.event_data['M1']['trigger_pattern'] == DATA_TRIGGER_PATTERN)[0]
                m2_ids_data = np.where(self.event_data['M2']['trigger_pattern'] == DATA_TRIGGER_PATTERN)[0]

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
                m1_ids_data = np.where(self.event_data['M1']['trigger_pattern'] == DATA_TRIGGER_PATTERN)[0]
                mono_ids['M1'] = m1_ids_data.tolist()
            elif (n_m1_events == 0) and (n_m2_events != 0):
                m2_ids_data = np.where(self.event_data['M2']['trigger_pattern'] == DATA_TRIGGER_PATTERN)[0]
                mono_ids['M2'] = m2_ids_data.tolist()
        else:
            # just find ids where event stereo number is 0 (which is given to mono events) and pattern is MC trigger
            m1_mono_mask = np.logical_and(self.event_data['M1']['trigger_pattern'] == MC_TRIGGER_PATTERN, self.event_data['M1']['stereo_event_number'] == 0)
            m2_mono_mask = np.logical_and(self.event_data['M2']['trigger_pattern'] == MC_TRIGGER_PATTERN, self.event_data['M2']['stereo_event_number'] == 0)

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
            'mjd' - event arrival time [MJD]
        """

        file_num = self._get_pedestal_file_num(pedestal_event_num, telescope)
        event_id = self.pedestal_ids[telescope][pedestal_event_num]

        id_in_file = event_id - \
            self.event_data[telescope]['file_edges'][file_num]

        photon_content = self.event_data[telescope]['charge'][file_num][id_in_file][:self.n_camera_pixels]
        arrival_times = self.event_data[telescope]['arrival_time'][file_num][id_in_file][:self.n_camera_pixels]

        event_data = dict()
        event_data['image'] = np.array(photon_content)
        event_data['pulse_time'] = np.array(arrival_times)
        event_data['pointing_az'] = self.event_data[telescope]['pointing_az'][event_id]
        event_data['pointing_zd'] = self.event_data[telescope]['pointing_zd'][event_id]
        event_data['pointing_ra'] = self.event_data[telescope]['pointing_ra'][event_id]
        event_data['pointing_dec'] = self.event_data[telescope]['pointing_dec'][event_id]
        event_data['mjd'] = self.event_data[telescope]['MJD'][event_id]
        event_data['mars_meta'] = self.event_data[telescope]['mars_meta'][file_num]

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
            'mjd' - event arrival time [MJD]
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

        # get information identical for both telescopes from M1:
        event_data['mars_meta'] = self.event_data['M1']['mars_meta'][m1_file_num]

        if not self.is_mc:
            event_data['mjd'] = self.event_data['M1']['MJD'][m1_id]
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
            'mjd' - event arrival time [MJD]
        """

        file_num = self._get_mono_file_num(mono_event_num, telescope)
        event_id = self.mono_ids[telescope][mono_event_num]

        id_in_file = event_id - \
            self.event_data[telescope]['file_edges'][file_num]

        photon_content = self.event_data[telescope]['charge'][file_num][id_in_file][:self.n_camera_pixels]
        arrival_times = self.event_data[telescope]['arrival_time'][file_num][id_in_file][:self.n_camera_pixels]

        event_data = dict()
        event_data['image'] = photon_content
        event_data['pulse_time'] = arrival_times
        event_data['pointing_az'] = self.event_data[telescope]['pointing_az'][event_id]
        event_data['pointing_zd'] = self.event_data[telescope]['pointing_zd'][event_id]
        event_data['pointing_ra'] = self.event_data[telescope]['pointing_ra'][event_id]
        event_data['pointing_dec'] = self.event_data[telescope]['pointing_dec'][event_id]

        event_data['mars_meta'] = self.event_data[telescope]['mars_meta'][file_num]

        if not self.is_mc:
            event_data['mjd'] = self.event_data[telescope]['MJD'][event_id]
        else:
            event_data['true_energy'] = self.event_data[telescope]['true_energy'][event_id]
            event_data['true_zd'] = self.event_data[telescope]['true_zd'][event_id]
            event_data['true_az'] = self.event_data[telescope]['true_az'][event_id]
            event_data['true_shower_primary_id'] = self.event_data[telescope]['true_shower_primary_id'][event_id]
            event_data['true_h_first_int'] = self.event_data[telescope]['true_h_first_int'][event_id]
            event_data['true_core_x'] = self.event_data[telescope]['true_core_x'][event_id]
            event_data['true_core_y'] = self.event_data[telescope]['true_core_y'][event_id]

        return event_data

class MarsMelibeaRun:
    """
    This class implements reading of the event data from a single MAGIC melibea run.
    """

    def __init__(self, run_file_mask, filter_list=None):
        """
        Constructor of the class. Defines the run to use and the camera pixel arrangement.

        Parameters
        ----------
        run_file_mask: str
            A path mask for files belonging to the run. Must correspond to a single run
            or an exception will be raised. Must correspond to calibrated ("sorcerer"-level)
            data.
        filter_list: list, optional
            A list of files, to which the run_file_mask should be applied. If None, all the
            files satisfying run_file_mask will be used. Defaults to None.
        """

        self.n_camera_pixels = GEOM.n_pixels

        self.run_file_mask = run_file_mask

        # Preparing the lists of M1/2 data files
        file_list = glob.glob(run_file_mask)

        # Filtering out extra files if necessary
        if filter_list is not None:
            file_list = list(set(file_list) & set(filter_list))

        # Retrieving the list of run numbers corresponding to the data files
        run_info = list(
            map(MAGICEventSource.get_run_info_from_name, file_list))
        is_mc_runs = [i[1] for i in run_info]

        is_mc_runs = np.unique(is_mc_runs)
        # Checking if run type (data/MC) is consistent:
        if len(is_mc_runs) > 1:
            raise ValueError(
                "Run type is not consistently data or MC: {}".format(is_mc_runs))

        self.is_mc = is_mc_runs[0]

        # Reading the data
        m1_data, m2_data, stereo_data = self.load_events(file_list, self.is_mc)

        # Getting the event data
        self.event_data = dict()
        self.event_data['M1'] = m1_data
        self.event_data['M2'] = m2_data
        self.event_data['stereo'] = stereo_data

        # Detecting stereo events
        self.stereo_ids = self._find_stereo_events()

    @property
    def n_events_m1(self):
        return len(self.event_data['M1']['MJD'])

    @property
    def n_events_m2(self):
        return len(self.event_data['M2']['MJD'])

    @property
    def n_stereo_events(self):
        return len(self.event_data['stereo'])

    @staticmethod
    def load_events(file_list, is_mc):
        """
        This method loads events and monitoring data from the pre-defiled file and returns them as a dictionary.

        Parameters
        ----------
        file_name: str
            Name of the MAGIC calibrated file to use.
        is_mc: boolean
            Specify whether Monte Carlo (True) or data (False) events are read

        Returns
        -------
        dict:
            A dictionary with the even properties: charge / arrival time data, trigger, direction etc.
        """

        try:
            import uproot
        except ImportError:
            msg = "The `uproot` python module is required to access the MAGIC data"
            raise ImportError(msg)

        event_data_m1     = dict()
        event_data_m2     = dict()
        event_data_stereo = dict()

        event_data_m1['stereo_event_number'] = np.array([], dtype=np.int32)
        event_data_m1['pointing_zd'] = np.array([])
        event_data_m1['pointing_az'] = np.array([])
        event_data_m1['pointing_ra'] = np.array([])
        event_data_m1['pointing_dec'] = np.array([])
        event_data_m1['MJD'] = np.array([])
        event_data_m1['length']          = np.array([])
        event_data_m1['width']           = np.array([])
        event_data_m1['delta']           = np.array([])
        event_data_m1['intensity']       = np.array([])
        event_data_m1['cog_x']           = np.array([])
        event_data_m1['cog_y']           = np.array([])
        event_data_m1['num_islands']     = np.array([])
        event_data_m1['leakage1']        = np.array([])
        event_data_m1['leakage2']        = np.array([])
        event_data_m1['num_used_pixels'] = np.array([])
        event_data_m1['time_intercept']  = np.array([])
        event_data_m1['time_slope']      = np.array([])
        event_data_m1['energy_est'] = np.array([])
        event_data_m1['energy_est_rms'] = np.array([])
        event_data_m1['energy_est_uncertainty'] = np.array([])
        event_data_m1['impact'] = np.array([])
        event_data_m1['disp_x'] = np.array([])
        event_data_m1['disp_y'] = np.array([])
        event_data_m1['disp_source_x'] = np.array([])
        event_data_m1['disp_source_y'] = np.array([])
        event_data_m1['mars_meta'] = []

        event_data_m2['stereo_event_number'] = np.array([], dtype=np.int32)
        event_data_m2['pointing_zd'] = np.array([])
        event_data_m2['pointing_az'] = np.array([])
        event_data_m2['pointing_ra'] = np.array([])
        event_data_m2['pointing_dec'] = np.array([])
        event_data_m2['MJD'] = np.array([])
        event_data_m2['length']          = np.array([])
        event_data_m2['width']           = np.array([])
        event_data_m2['delta']           = np.array([])
        event_data_m2['intensity']       = np.array([])
        event_data_m2['cog_x']           = np.array([])
        event_data_m2['cog_y']           = np.array([])
        event_data_m2['num_islands']     = np.array([])
        event_data_m2['leakage1']        = np.array([])
        event_data_m2['leakage2']        = np.array([])
        event_data_m2['num_used_pixels'] = np.array([])
        event_data_m2['time_intercept']  = np.array([])
        event_data_m2['time_slope']      = np.array([])
        event_data_m2['energy_est'] = np.array([])
        event_data_m2['energy_est_rms'] = np.array([])
        event_data_m2['energy_est_uncertainty'] = np.array([])
        event_data_m2['impact'] = np.array([])
        event_data_m2['disp_x'] = np.array([])
        event_data_m2['disp_y'] = np.array([])
        event_data_m2['disp_source_x'] = np.array([])
        event_data_m2['disp_source_y'] = np.array([])
        event_data_m2['mars_meta'] = []

        event_data_stereo['hmax'] = np.array([])
        event_data_stereo['is_valid'] = np.array([])
        event_data_stereo['alt'] = np.array([])
        event_data_stereo['az'] = np.array([])
        event_data_stereo['core_x'] = np.array([])
        event_data_stereo['core_y'] = np.array([])
        event_data_stereo['energy_est'] = np.array([])
        event_data_stereo['energy_est_rms'] = np.array([])
        event_data_stereo['energy_est_uncertainty'] = np.array([])
        event_data_stereo['hadronness'] = np.array([])

        # if no file in the list (e.g. when reading mono information), then simply
        # return empty dicts/array
        if len(file_list) == 0:
            return event_data_m1, event_data_m2, event_data_stereo

        degrees_per_hour = 15.0
        seconds_per_day = 86400.0
        seconds_per_hour = 3600.

        for file_name in file_list:

            input_file = uproot.open(file_name)

            for tel_num in [1,2]:

                evt_common_list = [
                    f'MRawEvtHeader_{tel_num}.fStereoEvtNumber',
                ]

                pointing_array_list = [
                    f'MPointingPos_{tel_num}.fZd',
                    f'MPointingPos_{tel_num}.fAz',
                    f'MPointingPos_{tel_num}.fRa',
                    f'MPointingPos_{tel_num}.fDec',
                    f'MPointingPos_{tel_num}.fDevZd',
                    f'MPointingPos_{tel_num}.fDevAz',
                    f'MPointingPos_{tel_num}.fDevHa',
                    f'MPointingPos_{tel_num}.fDevDec',
                ]

                # Info only applicable for data:
                time_array_list = [
                    f'MTime_{tel_num}.fMjd',
                    f'MTime_{tel_num}.fTime.fMilliSec',
                    f'MTime_{tel_num}.fNanoSec',
                ]

                # Info only applicable for MC:
                mc_list = [
                    f'MMcEvt_{tel_num}.fEnergy',
                    f'MMcEvt_{tel_num}.fTheta',
                    f'MMcEvt_{tel_num}.fPhi',
                    f'MMcEvt_{tel_num}.fPartId',
                    f'MMcEvt_{tel_num}.fZFirstInteraction',
                    f'MMcEvt_{tel_num}.fCoreX',
                    f'MMcEvt_{tel_num}.fCoreY'
                ]

                # Metadata, currently not strictly required
                metainfo_array_list = [
                    f'MRawRunHeader_{tel_num}.fRunNumber',
                    f'MRawRunHeader_{tel_num}.fRunType',
                    f'MRawRunHeader_{tel_num}.fSubRunIndex',
                    f'MRawRunHeader_{tel_num}.fSourceRA',
                    f'MRawRunHeader_{tel_num}.fSourceDEC',
                    f'MRawRunHeader_{tel_num}.fTelescopeNumber'
                ]

                hillas_array_list = [
                    f'MHillas_{tel_num}.fLength',
                    f'MHillas_{tel_num}.fWidth',
                    f'MHillas_{tel_num}.fDelta',
                    f'MHillas_{tel_num}.fSize',
                    f'MHillas_{tel_num}.fMeanX',
                    f'MHillas_{tel_num}.fMeanY',
                ]

                leakage_array_list = [
                    f'MImagePar_{tel_num}.fNumIslands',
                    f'MNewImagePar_{tel_num}.fLeakage1',
                    f'MNewImagePar_{tel_num}.fLeakage2',
                    f'MNewImagePar_{tel_num}.fNumUsedPixels',
                ]

                timing_array_list = [
                    f'MHillasTimeFit_{tel_num}.fP1Const',
                    f'MHillasTimeFit_{tel_num}.fP1Grad',
                ]

                energy_array_list = [
                    f'MEnergyEst_{tel_num}.fEnergy',
                    f'MEnergyEst_{tel_num}.fEnergyRMS',
                    f'MEnergyEst_{tel_num}.fImpact',
                    f'MEnergyEst_{tel_num}.fUncertainty',
                ]

                disp_array_list = [
                    f'MImageParDisp_{tel_num}.fDisp',
                    f'MImageParDisp_{tel_num}.fDispRMS',
                    f'MImageParDisp_{tel_num}.fXdisp',
                    f'MImageParDisp_{tel_num}.fYdisp',
                ]

                stereo_array_list = [
                    'MStereoPar.fValid',
                    'MStereoPar.fDirectionZd',
                    'MStereoPar.fDirectionAz',
                    'MStereoPar.fCoreX',
                    'MStereoPar.fCoreY',
                    'MStereoPar.fMaxHeight',
                ]

                stereo_energy_array_list = [
                    'MEnergyEst.fEnergy',
                    'MEnergyEst.fEnergyRMS',
                    'MEnergyEst.fUncertainty',
                ]

                hadronness_array_list = [
                    'MHadronness.fHadronness',
                ]

                events = input_file['Events'].arrays(evt_common_list, library="np")

                stereo_event_number = events[f'MRawEvtHeader_{tel_num}.fStereoEvtNumber']

                # Reading run-wise meta information (same for all events in subrun):
                mars_meta = dict()

                mars_meta['is_simulation'] = is_mc

                # Reading event timing information:
                if not is_mc:
                    event_times = input_file['Events'].arrays(time_array_list, library="np")
                    # Computing the event arrival time

                    event_mjd = event_times[f'MTime_{tel_num}.fMjd']
                    event_millisec = event_times[f'MTime_{tel_num}.fTime.fMilliSec']
                    event_nanosec = event_times[f'MTime_{tel_num}.fNanoSec']

                    event_mjd = event_mjd + \
                        (event_millisec / 1e3 + event_nanosec / 1e9) / seconds_per_day
                    event_data['MJD'] = np.concatenate(
                        (event_data['MJD'], event_mjd))

                # try to read RunHeaders tree (soft fail if not present, to pass current tests)
                try:
                    meta_info = input_file['RunHeaders'].arrays(
                        metainfo_array_list, library="np")

                    mars_meta['origin'] = "MAGIC"
                    mars_meta['input_url'] = file_name

                    mars_meta['number'] = int(
                        meta_info[f'MRawRunHeader_{tel_num}.fRunNumber'][0])
                    mars_meta['number_subrun'] = int(
                        meta_info[f'MRawRunHeader_{tel_num}.fSubRunIndex'][0])
                    mars_meta['source_ra'] = meta_info[f'MRawRunHeader_{tel_num}.fSourceRA'][0] / \
                        seconds_per_hour * degrees_per_hour * u.deg
                    mars_meta['source_dec'] = meta_info[f'MRawRunHeader_{tel_num}.fSourceDEC'][0] / \
                        seconds_per_hour * u.deg

                    is_mc_check = int(meta_info[f'MRawRunHeader_{tel_num}.fRunType'][0])
                    if is_mc_check == 0:
                        is_mc_check = False
                    elif is_mc_check == 256:
                        is_mc_check = True
                    else:
                        msg = "Run type (Data or MC) of MAGIC data file not recognised."
                        LOGGER.error(msg)
                        raise ValueError(msg)
                    if is_mc_check != is_mc:
                        msg = "Inconsistent run type (data or MC) between file name and runheader content."
                        LOGGER.error(msg)
                        raise ValueError(msg)

                except KeyError:
                    LOGGER.warning(
                        "RunHeaders tree not present in file. Cannot read meta information - will assume it is a real data run.")
                    is_mc = False

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

                event_data['mars_meta'].append(mars_meta)
                event_data['stereo_event_number'] = np.concatenate(
                    (event_data['stereo_event_number'], stereo_event_number))
                event_data['pointing_zd'] = np.concatenate(
                    (event_data['pointing_zd'], pointing_zd))
                event_data['pointing_az'] = np.concatenate(
                    (event_data['pointing_az'], pointing_az))
                event_data['pointing_ra'] = np.concatenate(
                    (event_data['pointing_ra'], pointing_ra))
                event_data['pointing_dec'] = np.concatenate(
                    (event_data['pointing_dec'], pointing_dec))

                hillas_parameters  = input_file['Events'].arrays(hillas_array_list, library="np")
                leakage_parameters = input_file['Events'].arrays(leakage_array_list, library="np")
                timing_parameters  = input_file['Events'].arrays(timing_array_list, library="np")
                energy             = input_file['Events'].arrays(energy_array_list, library="np")
                disp               = input_file['Events'].arrays(disp_array_list, library="np")

                event_data['length']                 = hillas_parameters[f'MHillas_{tel_num}.fLength'] * u.mm
                event_data['width']                  = hillas_parameters[f'MHillas_{tel_num}.fWidth'] * u.mm
                event_data['delta']                  = hillas_parameters[f'MHillas_{tel_num}.fDelta'] * u.rad
                event_data['intensity']              = hillas_parameters[f'MHillas_{tel_num}.fSize']
                event_data['cog_x']                  = hillas_parameters[f'MHillas_{tel_num}.fMeanX'] * u.mm
                event_data['cog_y']                  = hillas_parameters[f'MHillas_{tel_num}.fMeanY'] * u.mm
                event_data['num_islands']            = leakage_parameters[f'MImagePar_{tel_num}.fNumIslands']
                event_data['leakage1']               = leakage_parameters[f'MNewImagePar_{tel_num}.fLeakage1']
                event_data['leakage2']               = leakage_parameters[f'MNewImagePar_{tel_num}.fLeakage2']
                event_data['num_used_pixels']        = leakage_parameters[f'MNewImagePar_{tel_num}.fNumUsedPixels']
                event_data['time_intercept']         = timing_parameters[f'MHillasTimeFit_{tel_num}.fP1Const']
                event_data['time_slope']             = timing_parameters[f'MHillasTimeFit_{tel_num}.fP1Grad'] * 1/u.m
                event_data['energy_est']             = energy[f'MEnergyEst_{tel_num}.fEnergy'] * u.GeV
                event_data['energy_est_rms']         = energy[f'MEnergyEst_{tel_num}.fEnergyRMS'] * u.GeV
                event_data['energy_est_uncertainty'] = energy[f'MEnergyEst_{tel_num}.fUncertainty'] * u.GeV
                event_data['impact']                 = energy[f'MEnergyEst_{tel_num}.fImpact'] * u.cm
                event_data['disp_x']                 = disp[f'MImageParDisp_{tel_num}.fDisp'] * u.deg
                event_data['disp_y']                 = disp[f'MImageParDisp_{tel_num}.fDispRMS'] * u.deg
                event_data['disp_source_x']          = disp[f'MImageParDisp_{tel_num}.fXdisp'] * u.deg
                event_data['disp_source_y']          = disp[f'MImageParDisp_{tel_num}.fYdisp'] * u.deg

                if is_mc:
                    mc_info = input_file['Events'].arrays(mc_list, library="np")
                    # N.B.: For MC, there is only one subrun, so do not need to 'append'
                    event_data['true_energy']            = mc_info[f'MMcEvt_{tel_num}.fEnergy']
                    event_data['true_zd']                = mc_info[f'MMcEvt_{tel_num}.fTheta']
                    event_data['true_az']                = mc_info[f'MMcEvt_{tel_num}.fPhi']
                    event_data['true_shower_primary_id'] = mc_info[f'MMcEvt_{tel_num}.fPartId']
                    event_data['true_h_first_int']       = mc_info[f'MMcEvt_{tel_num}.fZFirstInteraction']
                    event_data['true_core_x']            = mc_info[f'MMcEvt_{tel_num}.fCoreX']
                    event_data['true_core_y']            = mc_info[f'MMcEvt_{tel_num}.fCoreY']

                if tel_num == 1:
                    event_data_m1 = event_data_m1
                else:
                    event_data_m2 = event_data_m1

            stereo_parameters  = input_file['Events'].arrays(stereo_array_list, library="np")
            stereo_energy      = input_file['Events'].arrays(stereo_energy_array_list, library="np")
            hadronness         = input_file['Events'].arrays(hadronness_array_list, library="np")

            event_data_stereo['hmax']                   = stereo_parameters['MStereoPar.fValid']
            event_data_stereo['alt']                    = stereo_parameters['MStereoPar.fDirectionZd']
            event_data_stereo['az']                     = stereo_parameters['MStereoPar.fDirectionAz']
            event_data_stereo['core_x']                 = stereo_parameters['MStereoPar.fCoreX']
            event_data_stereo['core_y']                 = stereo_parameters['MStereoPar.fCoreY']
            event_data_stereo['is_valid']               = stereo_parameters['MStereoPar.fMaxHeight']
            event_data_stereo['energy_est']             = stereo_energy['MEnergyEst.fEnergy']
            event_data_stereo['energy_est_rms']         = stereo_energy['MEnergyEst.fEnergyRMS']
            event_data_stereo['energy_est_uncertainty'] = stereo_energy['MEnergyEst.fUncertainty']
            event_data_stereo['hadronness']             = hadronness['MHadronness.fHadronness']

        return event_data_m1, event_data_m2, event_data_stereo

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
