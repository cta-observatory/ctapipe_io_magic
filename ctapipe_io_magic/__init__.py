# Event source for MAGIC calibrated data files.
# Requires uproot package (https://github.com/scikit-hep/uproot).
import logging

import glob
import re

import scipy
import numpy as np
import scipy.interpolate

from astropy import units as u
from astropy.time import Time
from ctapipe.io.eventsource import EventSource
from ctapipe.io.containers import DataContainer, TelescopePointingContainer, WeatherContainer
from ctapipe.instrument import TelescopeDescription, SubarrayDescription, OpticsDescription, CameraGeometry

__all__ = ['MAGICEventSource']

logger = logging.getLogger(__name__)

class MAGICEventSource(EventSource):
    """
    EventSource for MAGIC calibrated data.

    This class operates with the MAGIC data run-wise. This means that the files
    corresponding to the same data run are loaded and processed together.
    """
    _count = 0

    def __init__(self, config=None, tool=None, **kwargs):
        """
        Constructor

        Parameters
        ----------
        config: traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            Set to None if no configuration to pass.
        tool: ctapipe.core.Tool
            Tool executable that is calling this component.
            Passes the correct logger to the component.
            Set to None if no Tool to pass.
        kwargs: dict
            Additional parameters to be passed.
            NOTE: The file mask of the data to read can be passed with
            the 'input_url' parameter.
        """

        try:
            import uproot
        except ImportError:
            msg = "The `uproot` python module is required to access the MAGIC data"
            self.log.error(msg)
            raise

        self.file_list = glob.glob(kwargs['input_url'])
        self.file_list.sort()

        # EventSource can not handle file wild cards as input_url
        # To overcome this we substitute the input_url with first file matching
        # the specified file mask.
        del kwargs['input_url']
        super().__init__(input_url=self.file_list[0], **kwargs)

        # Retrieving the list of run numbers corresponding to the data files
        run_info = list(map(self._get_run_info_from_name, self.file_list))
        run_numbers = [i[0] for i in run_info]
        is_mc_runs = [i[1] for i in run_info]

        self.run_numbers, indices = np.unique(run_numbers, return_index=True)
        is_mc_runs = [is_mc_runs[i] for i in indices]
        is_mc_runs = np.unique(is_mc_runs)
        # Checking if runt type (data/MC) is consistent:
        if len(is_mc_runs) > 1:
            raise ValueError("Loaded files contain data and MC runs. Please load only data OR Monte Carlos.")
        self.is_mc = is_mc_runs[0]

        # # Setting up the current run with the first run present in the data
        # self.current_run = self._set_active_run(run_number=0)
        self.current_run = None

        # MAGIC telescope positions in m wrt. to the center of CTA simulations
        self.magic_tel_positions = {
            1: [-27.24, -146.66, 50.00] * u.m,
            2: [-96.44, -96.77, 51.00] * u.m
        }
        # MAGIC telescope description
        optics = OpticsDescription.from_name('MAGIC')
        geom = CameraGeometry.from_name('MAGICCam')
        self.magic_tel_description = TelescopeDescription(name='MAGIC', tel_type='MAGIC', optics=optics, camera=geom)
        self.magic_tel_descriptions = {1: self.magic_tel_description, 2: self.magic_tel_description}
        self._subarray_info = SubarrayDescription('MAGIC', self.magic_tel_positions, self.magic_tel_descriptions)

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

        file_list = glob.glob(file_mask)

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
                    pass

            except ImportError:
                if re.match(r'.+_m\d_.+root', file_path.lower()) is None:
                    is_magic_root_file = False

        return is_magic_root_file

    @staticmethod
    def _get_run_info_from_name(file_name):
        """
        This internal method extracts the run number and 
        type (data/MC) from the specified file name.

        Parameters
        ----------
        file_name: str
            A file name to process.

        Returns
        -------
        int:
            A run number of the file.
        """

        mask_data = r".*\d+_M\d+_(\d+)\.\d+_Y_.*"
        mask_mc = r".*_M\d_za\d+to\d+_\d_(\d+)_Y_.*"
        mask_mc_alt = r".*_M\d_\d_(\d+)_.*"
        if len(re.findall(mask_data, file_name)) > 0:
            parsed_info = re.findall(mask_data, file_name)
            is_mc = False
        elif len(re.findall(mask_mc, file_name)) > 0:
            parsed_info = re.findall(mask_mc, file_name)
            is_mc = True
        else:
            parsed_info = re.findall(mask_mc_alt, file_name)
            is_mc = True

        try:
            run_number = int(parsed_info[0])
        except IndexError:
            raise IndexError('Can not identify the run number and type (data/MC) of the file {:s}'.format(file_name))

        return run_number, is_mc

    def _set_active_run(self, run_number):
        """
        This internal method sets the run that will be used for data loading.

        Parameters
        ----------
        run_number: int
            The run number to use.

        Returns
        -------
        MarsRun:
            The run to use
        """

        input_path = '/'.join(self.input_url.split('/')[:-1])
        this_run_mask = input_path + '/*{:d}*root'.format(run_number)

        run = dict()
        run['number'] = run_number
        run['read_events'] = 0
        run['data'] = MarsRun(run_file_mask=this_run_mask, filter_list=self.file_list)

        return run

    @property
    def subarray(self):
        return self._subarray_info

    def _generator(self):
        """
        The default event generator. Return the stereo event
        generator instance.

        Returns
        -------

        """

        return self._stereo_event_generator()

    def _stereo_event_generator(self):
        """
        Stereo event generator. Yields DataContainer instances, filled
        with the read event data.

        Returns
        -------

        """

        counter = 0

        # Data container - is initialized once, and data is replaced within it after each yield
        data = DataContainer()

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

            # Loop over the events
            for event_i in range(self.current_run['data'].n_stereo_events):
                # Event and run ids
                event_order_number = self.current_run['data'].stereo_ids[event_i][0]
                event_id = self.current_run['data'].event_data['M1']['stereo_event_number'][event_order_number]
                obs_id = self.current_run['number']

                # Reading event data
                event_data = self.current_run['data'].get_stereo_event_data(event_i)
                
                data.meta = event_data['mars_meta']

                # Event counter
                data.count = counter

                # Setting up the R0 container
                data.r0.obs_id = obs_id
                data.r0.event_id = event_id
                data.r0.tel.clear()

                # Setting up the R1 container
                data.r1.obs_id = obs_id
                data.r1.event_id = event_id
                data.r1.tel.clear()

                # Setting up the DL0 container
                data.dl0.obs_id = obs_id
                data.dl0.event_id = event_id
                data.dl0.tel.clear()

                # Filling the DL1 container with the event data
                for tel_i, tel_id in enumerate(tels_in_file):
                    # Creating the telescope pointing container
                    pointing = TelescopePointingContainer()
                    pointing.azimuth = np.deg2rad(event_data['{:s}_pointing_az'.format(tel_id)]) * u.rad
                    pointing.altitude = np.deg2rad(90 - event_data['{:s}_pointing_zd'.format(tel_id)]) * u.rad
                    pointing.ra = np.deg2rad(event_data['{:s}_pointing_ra'.format(tel_id)]) * u.rad
                    pointing.dec = np.deg2rad(event_data['{:s}_pointing_dec'.format(tel_id)]) * u.rad

                    # Adding the pointing container to the event data
                    data.pointing[tel_i + 1] = pointing

                    # Adding event charge and peak positions per pixel
                    data.dl1.tel[tel_i + 1].image = event_data['{:s}_image'.format(tel_id)]
                    data.dl1.tel[tel_i + 1].pulse_time = event_data['{:s}_pulse_time'.format(tel_id)]
                    data.dl1.tel[tel_i + 1].badpixels = event_data['{:s}_bad_pixels'.format(tel_id)]
                    # data.dl1.tel[i_tel + 1].badpixels = np.array(
                    #     file['dl1/tel' + str(i_tel + 1) + '/badpixels'], dtype=np.bool)


                if self.is_mc == False:
                    # Adding the event arrival time
                    time_tmp = Time(event_data['mjd'], scale='utc', format='mjd')
                    data.trig.gps_time = Time(time_tmp, format='unix', scale='utc', precision=9)
                    # Filling weather information
                    weather = WeatherContainer()
                    weather.air_temperature = event_data['air_temperature'] * u.deg_C
                    weather.air_pressure = event_data['air_pressure'] * u.hPa
                    weather.air_humidity = event_data['air_humidity']
                    data.weather = weather
                else:
                    data.mc.energy = event_data['true_energy'] * u.GeV
                    data.mc.alt = (90 - event_data['true_zd']) * u.deg
                    data.mc.az = event_data['true_az'] * u.deg
                    data.mc.shower_primary_id = 1 - event_data['true_shower_primary_id']
                    data.mc.h_first_int = event_data['true_h_first_int'] * u.cm
                    data.mc.core_x = event_data['true_core_x'] * u.cm
                    data.mc.core_y = event_data['true_core_y'] * u.cm

                # Setting the telescopes with data
                data.r0.tels_with_data = tels_with_data
                data.r1.tels_with_data = tels_with_data
                data.dl0.tels_with_data = tels_with_data
                data.trig.tels_with_trigger = tels_with_data
                

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
        data = DataContainer()

        # Telescopes with data:
        tels_in_file = ["M1", "M2"]

        if telescope not in tels_in_file:
            raise ValueError("Specified telescope {:s} is not in the allowed list {}".format(telescope, tels_in_file))

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
                
                data.meta = event_data['mars_meta']

                # Event counter
                data.count = counter

                # Setting up the R0 container
                data.r0.obs_id = obs_id
                data.r0.event_id = event_id
                data.r0.tel.clear()

                # Setting up the R1 container
                data.r1.obs_id = obs_id
                data.r1.event_id = event_id
                data.r1.tel.clear()

                # Setting up the DL0 container
                data.dl0.obs_id = obs_id
                data.dl0.event_id = event_id
                data.dl0.tel.clear()

                # Creating the telescope pointing container
                pointing = TelescopePointingContainer()
                pointing.azimuth = np.deg2rad(event_data['pointing_az']) * u.rad
                pointing.altitude = np.deg2rad(90 - event_data['pointing_zd']) * u.rad
                pointing.ra = np.deg2rad(event_data['pointing_ra']) * u.rad
                pointing.dec = np.deg2rad(event_data['pointing_dec']) * u.rad

                # Adding the pointing container to the event data
                data.pointing[tel_i + 1] = pointing

                # Adding event charge and peak positions per pixel
                data.dl1.tel[tel_i + 1].image = event_data['image']
                data.dl1.tel[tel_i + 1].pulse_time = event_data['pulse_time']
                data.dl1.tel[tel_i + 1].badpixels = event_data['bad_pixels']
                # data.dl1.tel[tel_i + 1].badpixels = np.array(
                #     file['dl1/tel' + str(i_tel + 1) + '/badpixels'], dtype=np.bool)

                if self.is_mc == False:
                    # Adding the event arrival time
                    time_tmp = Time(event_data['mjd'], scale='utc', format='mjd')
                    data.trig.gps_time = Time(time_tmp, format='unix', scale='utc', precision=9)
                    # Filling weather information
                    weather = WeatherContainer()
                    weather.air_temperature = event_data['air_temperature'] * u.deg_C
                    weather.air_pressure = event_data['air_pressure'] * u.hPa
                    weather.air_humidity = event_data['air_humidity']
                    data.weather = weather
                else:
                    data.mc.energy = event_data['true_energy'] * u.GeV
                    data.mc.alt = (90 - event_data['true_zd']) * u.deg
                    data.mc.az = event_data['true_az'] * u.deg
                    data.mc.shower_primary_id = 1 - event_data['true_shower_primary_id']
                    data.mc.h_first_int = event_data['true_h_first_int'] * u.cm
                    data.mc.core_x = event_data['true_core_x'] * u.cm
                    data.mc.core_y = event_data['true_core_y'] * u.cm

                # Setting the telescopes with data
                data.r0.tels_with_data = tels_with_data
                data.r1.tels_with_data = tels_with_data
                data.dl0.tels_with_data = tels_with_data
                data.trig.tels_with_trigger = tels_with_data

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
        data = DataContainer()

        # Telescopes with data:
        tels_in_file = ["M1", "M2"]

        if telescope not in tels_in_file:
            raise ValueError("Specified telescope {:s} is not in the allowed list {}".format(telescope, tels_in_file))

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
                event_data = self.current_run['data'].get_pedestal_event_data(event_i, telescope=telescope)
                
                data.meta = event_data['mars_meta']

                # Event counter
                data.count = counter

                # Setting up the R0 container
                data.r0.obs_id = obs_id
                data.r0.event_id = event_id
                data.r0.tel.clear()

                # Setting up the R1 container
                data.r1.obs_id = obs_id
                data.r1.event_id = event_id
                data.r1.tel.clear()

                # Setting up the DL0 container
                data.dl0.obs_id = obs_id
                data.dl0.event_id = event_id
                data.dl0.tel.clear()

                # Creating the telescope pointing container
                pointing = TelescopePointingContainer()
                pointing.azimuth = np.deg2rad(event_data['pointing_az']) * u.rad
                pointing.altitude = np.deg2rad(90 - event_data['pointing_zd']) * u.rad
                pointing.ra = np.deg2rad(event_data['pointing_ra']) * u.rad
                pointing.dec = np.deg2rad(event_data['pointing_dec']) * u.rad

                # Adding the pointing container to the event data
                data.pointing[tel_i + 1] = pointing

                # Adding event charge and peak positions per pixel
                data.dl1.tel[tel_i + 1].image = event_data['image']
                data.dl1.tel[tel_i + 1].pulse_time = event_data['pulse_time']
                data.dl1.tel[tel_i + 1].badpixels = event_data['bad_pixels']
                # data.dl1.tel[tel_i + 1].badpixels = np.array(
                #     file['dl1/tel' + str(i_tel + 1) + '/badpixels'], dtype=np.bool)

                # Adding the event arrival time
                time_tmp = Time(event_data['mjd'], scale='utc', format='mjd')
                data.trig.gps_time = Time(time_tmp, format='unix', scale='utc', precision=9)

                # Setting the telescopes with data
                data.r0.tels_with_data = tels_with_data
                data.r1.tels_with_data = tels_with_data
                data.dl0.tels_with_data = tels_with_data
                data.trig.tels_with_trigger = tels_with_data
                
                if self.is_mc == False:
                    # Filling weather information
                    weather = WeatherContainer()
                    weather.air_temperature = event_data['air_temperature'] * u.deg_C
                    weather.air_pressure = event_data['air_pressure'] * u.hPa
                    weather.air_humidity = event_data['air_humidity']
                    data.weather = weather

                yield data
                counter += 1

        return


class MarsRun:
    """
    This class implements reading of the event data from a single MAGIC data run.
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

        self.run_file_mask = run_file_mask

        # Preparing the lists of M1/2 data files
        file_list = glob.glob(run_file_mask)

        # Filtering out extra files if necessary
        if filter_list is not None:
            file_list = list(set(file_list) & set(filter_list))

        self.m1_file_list = list(filter(lambda name: '_M1_' in name, file_list))
        self.m2_file_list = list(filter(lambda name: '_M2_' in name, file_list))
        self.m1_file_list.sort()
        self.m2_file_list.sort()

        # Retrieving the list of run numbers corresponding to the data files
        run_info = list(map(MAGICEventSource._get_run_info_from_name, file_list))
        run_numbers = [i[0] for i in run_info]
        is_mc_runs   = [i[1] for i in run_info]

        run_numbers = np.unique(run_numbers)
        is_mc_runs = np.unique(is_mc_runs)
        # Checking if runt type (data/MC) is consistent:
        if len(is_mc_runs) > 1:
            raise ValueError("Run type is not consistently data or MC: {}".format(is_mc))
        
        self.is_mc = is_mc_runs[0]

        # Checking if a single run is going to be read
        if len(run_numbers) > 1:
            raise ValueError("Run mask corresponds to more than one run: {}".format(run_numbers))

        # Reading the event data
        self.event_data = dict()
        self.event_data['M1'] = self.load_events(self.m1_file_list, self.is_mc)
        self.event_data['M2'] = self.load_events(self.m2_file_list, self.is_mc)

        # Detecting pedestal events
        self.pedestal_ids = self._find_pedestal_events()
        # Detecting stereo events
        self.stereo_ids = self._find_stereo_events()
        # Detecting mono events
        self.mono_ids = self._find_mono_events()

        self.n_camera_pixels = 1039

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
    def load_events(file_list, is_mc):
        """
        This method loads events from the pre-defiled file and returns them as a dictionary.

        Parameters
        ----------
        file_name: str
            Name of the MAGIC calibrated file to use.

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
        event_data['trigger_pattern'] = scipy.array([])
        event_data['stereo_event_number'] = scipy.array([])
        event_data['pointing_zd'] = scipy.array([])
        event_data['pointing_az'] = scipy.array([])
        event_data['pointing_ra'] = scipy.array([])
        event_data['pointing_dec'] = scipy.array([])
        event_data['MJD'] = scipy.array([])
        event_data['air_pressure'] = scipy.array([])
        event_data['air_humidity'] = scipy.array([])
        event_data['air_temperature'] = scipy.array([])
        event_data['badpixelinfo'] = []
        event_data['mars_meta'] = []

        # run-wise meta information (same for all events)
        mars_meta = dict()
        
        event_data['file_edges'] = [0]

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
        
        weather_array_list = [
            'MTimeWeather.fMjd',
            'MTimeWeather.fTime.fMilliSec',
            'MTimeWeather.fNanoSec',
            'MReportWeather.fPressure',
            'MReportWeather.fHumidity',
            'MReportWeather.fTemperature'
            ]

        # Info only applicable for MC:
        mc_list = [
            'MMcEvt.fEnergy',
            'MMcEvt.fTheta',
            'MMcEvt.fPhi',
            'MMcEvt.fPartId',
            'MMcEvt.fZFirstInteraction',
            'MMcEvt.fCoreX',
            'MMcEvt.fCoreY', 
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

            events = input_file['Events'].arrays(evt_common_list)

            # Reading the info common to MC and real data
            charge = events[b'MCerPhotEvt.fPixels.fPhot']
            arrival_time = events[b'MArrivalTime.fData']
            trigger_pattern = events[b'MTriggerPattern.fPrescaled']
            stereo_event_number = events[b'MRawEvtHeader.fStereoEvtNumber']

            # Reading meta information:
            mars_meta['is_simulation'] = is_mc
            try:
                meta_info = input_file['RunHeaders'].arrays(metainfo_array_list)
                
                mars_meta['origin'] = "MAGIC"
                mars_meta['input_url'] = file_name
    
                mars_meta['number'] = int(meta_info[b'MRawRunHeader.fRunNumber'][0])
                #mars_meta['number_subrun'] = int(meta_info[b'MRawRunHeader.fSubRunIndex'][0])
                mars_meta['source_ra'] = meta_info[b'MRawRunHeader.fSourceRA'][0] / seconds_per_hour * degrees_per_hour * u.deg
                mars_meta['source_dec'] = meta_info[b'MRawRunHeader.fSourceDEC'][0] / seconds_per_hour * u.deg
    
                is_simulation = int(meta_info[b'MRawRunHeader.fRunType'][0])
                if is_simulation == 0:
                    is_simulation = False
                elif is_simulation == 256:
                    is_simulation = True
                else:
                    msg = "Run type (Data or MC) of MAGIC data file not recognised."
                    logger.error(msg)
                    raise ValueError(msg)
                if is_simulation != is_mc:
                    msg = "Inconsistent run type (data or MC) between file name and runheader content."
                    logger.error(msg)
                    raise ValueError(msg)
                
                # Reading the info only contained in real data
                if is_simulation == False:
                    badpixelinfo = input_file['RunHeaders']['MBadPixelsCam.fArray.fInfo'].array(uproot.asjagged(uproot.asdtype(np.int32))).flatten().reshape((4, 1183), order='F')
                    # now we have 3 axes:
                    # 1st axis: Unsuitable pixels
                    # 2nd axis: Uncalibrated pixels (says why pixel is unsuitable)
                    # 3rd axis: Bad hardware pixels (says why pixel is unsuitable)
                    # Each axis cointains a 32bit integer encoding more information about the specific problem, see MARS software, MBADPixelsPix.h
                    # Here, we however discard this additional information and only grep the "unsuitable" axis.
                    badpixelinfo = badpixelinfo[1].astype(bool)
                else:
                    badpixelinfo = np.zeros(1183).astype(bool)
            except KeyError:
                logger.warning("RunHeaders tree not present in file. Cannot read meta information and assume it is a real data run.")
                badpixelinfo = np.zeros(1183)
                is_simulation = False

            if is_simulation == False:
                event_times = input_file['Events'].arrays(time_array_list)
                # Computing the event arrival time
                
                mjd = event_times[b'MTime.fMjd']
                millisec = event_times[b'MTime.fTime.fMilliSec']
                nanosec = event_times[b'MTime.fNanoSec']
    
                mjd = mjd + (millisec / 1e3 + nanosec / 1e9) / seconds_per_day
    
                # Reading weather information:
                try:
                    weather_info = input_file['Weather'].arrays(weather_array_list)
                    
                    weather_time_day = weather_info[b'MTimeWeather.fMjd']
                    weather_time_millisec = weather_info[b'MTimeWeather.fTime.fMilliSec']
                    weather_time_nanosec = weather_info[b'MTimeWeather.fNanoSec']
                    weather_mjd = weather_time_day + (weather_time_millisec/1e3 + weather_time_nanosec/1e9) / seconds_per_day
                    weather_mjd, weather_indices = np.unique(weather_mjd, return_index = True)
                    
                    air_pressure_array = weather_info[b'MReportWeather.fPressure'][weather_indices] # hPa
                    air_humidity_array = weather_info[b'MReportWeather.fHumidity'][weather_indices]
                    air_temperature_array = weather_info[b'MReportWeather.fTemperature'][weather_indices] # degree celsius
          
                    air_pressure_interpolator = scipy.interpolate.interp1d(weather_mjd, air_pressure_array, fill_value="extrapolate")
                    air_humidity_interpolator = scipy.interpolate.interp1d(weather_mjd, air_humidity_array, fill_value="extrapolate")
                    air_temperature_interpolator = scipy.interpolate.interp1d(weather_mjd, air_temperature_array, fill_value="extrapolate")
                      
                    air_pressure = air_pressure_interpolator(mjd) #* u.hPa
                    air_humidity = air_humidity_interpolator(mjd)
                    air_temperature = air_temperature_interpolator(mjd) #* u.deg_C
                except:
                    print("Could not find weather information. "
                                 "Set to 0 degree Celsius, 50% humidity, 790hPa ambient pressure.")
                    air_pressure = scipy.full(len(mjd), 790.) #* u.hPa
                    air_humidity = scipy.full(len(mjd), 0.5)
                    air_temperature = scipy.zeros(len(mjd)) #* u.deg_C

            # Reading pointing information (in units of degrees):
            if 'MPointingPos.' in input_file['Events']:
                # Retrieving the telescope pointing direction
                pointing = input_file['Events'].arrays(pointing_array_list)

                pointing_zd = pointing[b'MPointingPos.fZd'] - pointing[b'MPointingPos.fDevZd']
                pointing_az = pointing[b'MPointingPos.fAz'] - pointing[b'MPointingPos.fDevAz']
                pointing_ra = (pointing[b'MPointingPos.fRa'] + pointing[b'MPointingPos.fDevHa']) * degrees_per_hour # N.B. the positive sign here, as HA = local sidereal time - ra
                pointing_dec = pointing[b'MPointingPos.fDec'] - pointing[b'MPointingPos.fDevDec']
            else:
                # Getting the telescope drive info
                drive = input_file['Drive'].arrays(drive_array_list)

                drive_mjd = drive[b'MReportDrive.fMjd']
                drive_zd = drive[b'MReportDrive.fCurrentZd']
                drive_az = drive[b'MReportDrive.fCurrentAz']
                drive_ra = drive[b'MReportDrive.fRa'] * degrees_per_hour
                drive_dec = drive[b'MReportDrive.fDec']

                # Finding only non-repeating drive entries
                # Repeating entries lead to failure in pointing interpolation
                non_repeating = scipy.diff(drive_mjd) > 0
                non_repeating = scipy.concatenate((non_repeating, [True]))

                # Filtering out the repeating ones
                drive_mjd = drive_mjd[non_repeating]
                drive_zd = drive_zd[non_repeating]
                drive_az = drive_az[non_repeating]
                drive_ra = drive_ra[non_repeating]
                drive_dec = drive_dec[non_repeating]

                if len(drive_zd) > 2:
                    # If there are enough drive data, creating azimuth and zenith angles interpolators
                    drive_zd_pointing_interpolator = scipy.interpolate.interp1d(drive_mjd, drive_zd, fill_value="extrapolate")
                    drive_az_pointing_interpolator = scipy.interpolate.interp1d(drive_mjd, drive_az, fill_value="extrapolate")

                    # Creating azimuth and zenith angles interpolators
                    drive_ra_pointing_interpolator = scipy.interpolate.interp1d(drive_mjd, drive_ra, fill_value="extrapolate")
                    drive_dec_pointing_interpolator = scipy.interpolate.interp1d(drive_mjd, drive_dec, fill_value="extrapolate")

                    # Interpolating the drive pointing to the event time stamps
                    pointing_zd = drive_zd_pointing_interpolator(mjd)
                    pointing_az = drive_az_pointing_interpolator(mjd)
                    pointing_ra = drive_ra_pointing_interpolator(mjd)
                    pointing_dec = drive_dec_pointing_interpolator(mjd)

                else:
                    # Not enough data to interpolate the pointing direction.
                    pointing_zd = scipy.repeat(-1, len(mjd))
                    pointing_az = scipy.repeat(-1, len(mjd))
                    pointing_ra = scipy.repeat(-1, len(mjd))
                    pointing_dec = scipy.repeat(-1, len(mjd))

            event_data['charge'].append(charge)
            event_data['arrival_time'].append(arrival_time)
            event_data['badpixelinfo'].append(badpixelinfo)
            event_data['mars_meta'].append(mars_meta)
            event_data['trigger_pattern'] = scipy.concatenate((event_data['trigger_pattern'], trigger_pattern))
            event_data['stereo_event_number'] = scipy.concatenate((event_data['stereo_event_number'], stereo_event_number)).astype(dtype='int')
            event_data['pointing_zd'] = scipy.concatenate((event_data['pointing_zd'], pointing_zd))
            event_data['pointing_az'] = scipy.concatenate((event_data['pointing_az'], pointing_az))
            event_data['pointing_ra'] = scipy.concatenate((event_data['pointing_ra'], pointing_ra))
            event_data['pointing_dec'] = scipy.concatenate((event_data['pointing_dec'], pointing_dec))
            if is_simulation == False:
                event_data['air_pressure'] = scipy.concatenate((event_data['air_pressure'], air_pressure))
                event_data['air_humidity'] = scipy.concatenate((event_data['air_humidity'], air_humidity))
                event_data['air_temperature'] = scipy.concatenate((event_data['air_temperature'], air_temperature))
    
                event_data['MJD'] = scipy.concatenate((event_data['MJD'], mjd))
            else:
                mc_info = input_file['Events'].arrays(mc_list)
                # N.B.: For MC, there is only one subrun
                event_data['true_energy'] = mc_info[b'MMcEvt.fEnergy']
                event_data['true_zd'] = mc_info[b'MMcEvt.fTheta']
                event_data['true_az'] = mc_info[b'MMcEvt.fPhi']
                event_data['true_shower_primary_id'] = mc_info[b'MMcEvt.fPartId']
                event_data['true_h_first_int'] = mc_info[b'MMcEvt.fZFirstInteraction']
                event_data['true_core_x'] = mc_info[b'MMcEvt.fCoreX']
                event_data['true_core_y'] = mc_info[b'MMcEvt.fCoreY']

            event_data['file_edges'].append(len(event_data['trigger_pattern']))

        return event_data

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

        pedestal_trigger_pattern = 8

        for telescope in self.event_data:
            ped_triggers = np.where(self.event_data[telescope]['trigger_pattern'] == pedestal_trigger_pattern)
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
            data_trigger_pattern = 128

            m2_data_condition = (self.event_data['M2']['trigger_pattern'] == data_trigger_pattern)

            for m1_id in range(0, n_m1_events):
                if self.event_data['M1']['trigger_pattern'][m1_id] == data_trigger_pattern:
                    m2_stereo_condition = (self.event_data['M2']['stereo_event_number'] ==
                                           self.event_data['M1']['stereo_event_number'][m1_id])

                    m12_match = np.where(m2_data_condition & m2_stereo_condition)

                    if len(m12_match[0]) > 0:
                        stereo_pair = (m1_id, m12_match[0][0])
                        stereo_ids.append(stereo_pair)
        else:
            data_trigger_pattern = 1

            m2_data_condition = (self.event_data['M2']['trigger_pattern'] == data_trigger_pattern)

            for m1_id in range(0, n_m1_events):
                if self.event_data['M1']['trigger_pattern'][m1_id] == data_trigger_pattern and self.event_data['M1']['stereo_event_number'][m1_id] != 0:
                    m2_stereo_condition = (self.event_data['M2']['stereo_event_number'] ==
                                           self.event_data['M1']['stereo_event_number'][m1_id])

                    m12_match = np.where(m2_data_condition & m2_stereo_condition)

                    if len(m12_match[0]) > 0:
                        stereo_pair = (m1_id, m12_match[0][0])
                        stereo_ids.append(stereo_pair)

        return stereo_ids

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
            data_trigger_pattern = 128
    
            m1_data_condition = self.event_data['M1']['trigger_pattern'] == data_trigger_pattern
            m2_data_condition = self.event_data['M2']['trigger_pattern'] == data_trigger_pattern
    
            for m1_id in range(0, n_m1_events):
                if m1_data_condition[m1_id]:
                    m2_stereo_condition = (self.event_data['M2']['stereo_event_number'] ==
                                           self.event_data['M1']['stereo_event_number'][m1_id])
    
                    m12_match = np.where(m2_data_condition & m2_stereo_condition)
    
                    if len(m12_match[0]) == 0:
                        mono_ids['M1'].append(m1_id)
    
            for m2_id in range(0, n_m2_events):
                if m2_data_condition[m2_id]:
                    m1_stereo_condition = (self.event_data['M1']['stereo_event_number'] ==
                                           self.event_data['M2']['stereo_event_number'][m2_id])
    
                    m12_match = np.where(m1_data_condition & m1_stereo_condition)
    
                    if len(m12_match[0]) == 0:
                        mono_ids['M2'].append(m2_id)
        else:

            data_trigger_pattern = 1
            m1_data_condition = self.event_data['M1']['trigger_pattern'] == data_trigger_pattern
            m2_data_condition = self.event_data['M2']['trigger_pattern'] == data_trigger_pattern

            # shortcut if only single file is loaded:
            if n_m1_events == 0:
                mono_ids['M2'] = np.arange(0,n_m2_events)[m2_data_condition]
                return mono_ids
            if n_m2_events == 0:
                mono_ids['M1'] = np.arange(0,n_m1_events)[m1_data_condition]
                return mono_ids

            for m1_id in range(0, n_m1_events):
                if m1_data_condition[m1_id]:
                    if self.event_data['M1']['stereo_event_number'][m1_id] == 0:
                        mono_ids['M1'].append(m1_id)
            for m2_id in range(0, n_m2_events):
                if m2_data_condition[m2_id]:
                    if self.event_data['M2']['stereo_event_number'][m2_id] == 0:
                        mono_ids['M2'].append(m2_id)

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
        file_num = np.digitize([event_id], self.event_data[telescope]['file_edges'])
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
        file_num = np.digitize([event_id], self.event_data[telescope]['file_edges'])
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
            'bad_pixels' - boolean array indicating problematic pixels
            'pointing_az' - pointing azimuth [degrees]
            'pointing_zd' - pointing zenith angle [degrees]
            'pointing_ra' - pointing right ascension [degrees]
            'pointing_dec' - pointing declination [degrees]
            'mjd' - event arrival time [MJD]
            'air_humidity' - relative ambient air humidity
            'air_pressure' - ambient air pressure [astropy units]
            'air_temperature' - ambient air temperature [astropy units]
        """

        file_num = self._get_pedestal_file_num(pedestal_event_num, telescope)
        event_id = self.pedestal_ids[telescope][pedestal_event_num]

        id_in_file = event_id - self.event_data[telescope]['file_edges'][file_num]

        photon_content = self.event_data[telescope]['charge'][file_num][id_in_file][:self.n_camera_pixels]
        arrival_times = self.event_data[telescope]['arrival_time'][file_num][id_in_file][:self.n_camera_pixels]
        bad_pixels = self.event_data[telescope]['badpixelinfo'][file_num][:self.n_camera_pixels]

        event_data = dict()
        event_data['image'] = photon_content
        event_data['pulse_time'] = arrival_times
        event_data['bad_pixels'] = bad_pixels
        event_data['pointing_az'] = self.event_data[telescope]['pointing_az'][event_id]
        event_data['pointing_zd'] = self.event_data[telescope]['pointing_zd'][event_id]
        event_data['pointing_ra'] = self.event_data[telescope]['pointing_ra'][event_id]
        event_data['pointing_dec'] = self.event_data[telescope]['pointing_dec'][event_id]
        event_data['mjd'] = self.event_data[telescope]['MJD'][event_id]
        event_data['air_pressure'] = self.event_data[telescope]['air_pressure'][event_id]
        event_data['air_humidity'] = self.event_data[telescope]['air_humidity'][event_id]
        event_data['air_temperature'] = self.event_data[telescope]['air_temperature'][event_id]               
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
            'm1_bad_pixels' - boolean array indicating problematic M1 pixels
            'm2_image' - M2 photon_content
            'm2_peak_pos' - M2 arrival_times
            'm2_bad_pixels' - boolean array indicating problematic M2 pixels
            'm1_pointing_az' - M1 pointing azimuth [degrees]
            'm1_pointing_zd' - M1 pointing zenith angle [degrees]
            'm1_pointing_ra' - M1 pointing right ascension [degrees]
            'm1_pointing_dec' - M1 pointing declination [degrees]
            'm2_pointing_az' - M2 pointing azimuth [degrees]
            'm2_pointing_zd' - M2 pointing zenith angle [degrees]
            'm2_pointing_ra' - M2 pointing right ascension [degrees]
            'm2_pointing_dec' - M2 pointing declination [degrees]
            'mjd' - event arrival time [MJD]
            'air_humidity' - relative ambient air humidity
            'air_pressure' - ambient air pressure [astropy units]
            'air_temperature' - ambient air temperature [astropy units]
        """

        m1_file_num, m2_file_num = self._get_stereo_file_num(stereo_event_num)
        m1_id = self.stereo_ids[stereo_event_num][0]
        m2_id = self.stereo_ids[stereo_event_num][1]

        m1_id_in_file = m1_id - self.event_data['M1']['file_edges'][m1_file_num]
        m2_id_in_file = m2_id - self.event_data['M2']['file_edges'][m2_file_num]

        m1_photon_content = self.event_data['M1']['charge'][m1_file_num][m1_id_in_file][:self.n_camera_pixels]
        m1_arrival_times = self.event_data['M1']['arrival_time'][m1_file_num][m1_id_in_file][:self.n_camera_pixels]
        m1_bad_pixels = self.event_data['M1']['badpixelinfo'][m1_file_num][:self.n_camera_pixels]

        m2_photon_content = self.event_data['M2']['charge'][m2_file_num][m2_id_in_file][:self.n_camera_pixels]
        m2_arrival_times = self.event_data['M2']['arrival_time'][m2_file_num][m2_id_in_file][:self.n_camera_pixels]
        m2_bad_pixels = self.event_data['M2']['badpixelinfo'][m2_file_num][:self.n_camera_pixels]

        event_data = dict()
        event_data['m1_image'] = m1_photon_content
        event_data['m1_pulse_time'] = m1_arrival_times
        event_data['m1_bad_pixels'] = m1_bad_pixels
        event_data['m2_image'] = m2_photon_content
        event_data['m2_pulse_time'] = m2_arrival_times
        event_data['m2_bad_pixels'] = m2_bad_pixels
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
        
        if self.is_mc == False:
            event_data['mjd'] = self.event_data['M1']['MJD'][m1_id]
            event_data['air_pressure'] = self.event_data['M1']['air_pressure'][m1_id]
            event_data['air_humidity'] = self.event_data['M1']['air_humidity'][m1_id]
            event_data['air_temperature'] = self.event_data['M1']['air_temperature'][m1_id]
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
            'bad_pixels' - boolean array indicating problematic pixels
            'pointing_az' - pointing azimuth [degrees]
            'pointing_zd' - pointing zenith angle [degrees]
            'pointing_ra' - pointing right ascension [degrees]
            'pointing_dec' - pointing declination [degrees]
            'mjd' - event arrival time [MJD]
            'air_humidity' - relative ambient air humidity
            'air_pressure' - ambient air pressure [astropy units]
            'air_temperature' - ambient air temperature [astropy units]
        """

        file_num = self._get_mono_file_num(mono_event_num, telescope)
        event_id = self.mono_ids[telescope][mono_event_num]

        id_in_file = event_id - self.event_data[telescope]['file_edges'][file_num]

        photon_content = self.event_data[telescope]['charge'][file_num][id_in_file][:self.n_camera_pixels]
        arrival_times = self.event_data[telescope]['arrival_time'][file_num][id_in_file][:self.n_camera_pixels]
        bad_pixels = self.event_data[telescope]['badpixelinfo'][file_num][:self.n_camera_pixels]

        event_data = dict()
        event_data['image'] = photon_content
        event_data['pulse_time'] = arrival_times
        event_data['bad_pixels'] = bad_pixels
        event_data['pointing_az'] = self.event_data[telescope]['pointing_az'][event_id]
        event_data['pointing_zd'] = self.event_data[telescope]['pointing_zd'][event_id]
        event_data['pointing_ra'] = self.event_data[telescope]['pointing_ra'][event_id]
        event_data['pointing_dec'] = self.event_data[telescope]['pointing_dec'][event_id]
        
        event_data['mars_meta'] = self.event_data[telescope]['mars_meta'][file_num]

        if self.is_mc == False:
            event_data['mjd'] = self.event_data[telescope]['MJD'][event_id]
            event_data['air_pressure'] = self.event_data[telescope]['air_pressure'][event_id]
            event_data['air_humidity'] = self.event_data[telescope]['air_humidity'][event_id]
            event_data['air_temperature'] = self.event_data[telescope]['air_temperature'][event_id]
        else:
            event_data['true_energy'] = self.event_data[telescope]['true_energy'][event_id]
            event_data['true_zd'] = self.event_data[telescope]['true_zd'][event_id]
            event_data['true_az'] = self.event_data[telescope]['true_az'][event_id]
            event_data['true_shower_primary_id'] = self.event_data[telescope]['true_shower_primary_id'][event_id]
            event_data['true_h_first_int'] = self.event_data[telescope]['true_h_first_int'][event_id]
            event_data['true_core_x'] = self.event_data[telescope]['true_core_x'][event_id]
            event_data['true_core_y'] = self.event_data[telescope]['true_core_y'][event_id]
        

        return event_data



