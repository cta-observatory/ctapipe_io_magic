## *ctapipe* MAGIC event source

EventSource plugin for *ctapipe*, needed to read the calibrated data of the MAGIC telescope system. It requires the [*ctapipe*](https://github.com/cta-observatory/ctapipe) (v0.12.0) and [*uproot*](https://github.com/scikit-hep/uproot4) (>=4.1) packages to run.

#### Installation

Provided that *ctapipe* is already installed, the installation can be done via *pip* (the module is available in PyPI):

```bash
pip install ctapipe_io_magic
```

Alternatively, you can always clone the repository and install like in the following:

```bash
git clone https://github.com/cta-observatory/ctapipe_io_magic.git
pip install ./ctapipe_io_magic/
```

This installation via *pip* (provided, *pip* is installed) has the advantage to be nicely controlled for belonging to a given conda environment (and to be uninstalled). Alternatively, do

```bash
git clone https://github.com/cta-observatory/ctapipe_io_magic.git
cd ctapipe_io_magic
python setup.py install --user
```

In all cases, using *pip* will check if the version of *ctapipe* and *uproot* is compatible with the requested version of *ctapipe_io_magic*.

#### Usage

```python
import ctapipe
from ctapipe_io_magic import MAGICEventSource

with MAGICEventSource(input_url=file_name) as event_source:
    for event in event_source:
        ...some processing...
```

With more recent versions of *ctapipe*, only one file at a time can be read. However, by default if a subrun of calibrated data is given as input, `MAGICEventSource` will read the events from all the subruns from the run to which the data file belongs. To suppress this behavior, set `process_run=False` No matching of the events is performed at this level (if stereo data).

By default, assuming a calibrated file as input, the event generator will generate:
-   if real data taken in stereo mode, cosmic events (trigger pattern = 128) from the corresponding telescope
-   if real data taken in mono mode (either as a single telescope or with both telescopes independently), cosmic events (trigger pattern = 1) from the corresponding telescope
-   if simulated data in stereo mode, cosmic events (trigger pattern = 1 and stereo trigger number different from 0) from the corresponding telescope

Pedestal events (trigger pattern = 8) and simulated events triggered by only one telescope (trigger pattern = 1 and stereo trigger number = 0) can be generated as well.

The reader is able to handle real data or Monte Carlo files, which are automatically recognized. Note that the names of input files have to follow the convention:
-   `*_M[1-2]_RUNNUMBER.SUBRUNNR_Y_*.root` for real data
-   `*_M[1-2]_za??to??_?_RUNNUMBER_Y_*.root` for simulated data.

However, the information which can be extracted from the file names is read directly from within the ROOT files.

##### More usage

Select a single run:

```python
run = event_source._set_active_run(event_source.run_numbers[0])
# run is an object of type MarsCalibratedRun
# assuming we are reading data from a file from M1
for n in range(run['data'].n_mono_events_m1):
    run['data'].get_mono_event_data(n, 'M1')

for n in range(run['data'].n_pedestal_events_m1):
    run['data'].get_pedestal_event_data(n, 'M1')
```

Select events triggering in stereo and pedestal events from a single telescope (recognized automatically) over event generator:

```python
# select (stereo) cosmic events from all subruns of a given run (the one to which file_name belongs)
event_generator = MAGICEventSource(input_url=file_name)
for cosmic_event in event_generator:
    ...some processing...

# select (stereo) cosmic events from a single subrun
event_generator = MAGICEventSource(input_url=file_name, process_run=False)
for cosmic_event in event_generator:
    ...some processing...

# select pedestal events
pedestal_event_generator = MAGICEventSource(input_url=file_name, use_pedestals=True)
for pedestal_event in pedestal_event_generator:
    ...some processing...
```

#### Features

##### Drive reports interpolation

By default, when all subruns from a given run are processed, the drive reports are collected from all subruns so that the telescope pointing position for each event can be computed. Also in the case that only one subrun is processed (`process_run=False`), all drive reports from the subruns belonging to the same run will be used. This ensures that interpolation is performed correctly.

##### Monitoring data

Monitoring data are saved in `run['data'].monitoring_data` and can also accessed event-wise via the `event.mon` container. Available information is:
-   dead pixels: `event.mon.tel[tel_id].pixel_status.hardware_failing_pixels[0]`
-   hot pixels:  `event.mon.tel[tel_id].pixel_status.pedestal_failing_pixels[i_ped_type]`, where `i_ped_type` is an index indicating which pedestal type to use (0 is `PedestalFundamental`, 1 is `PedestalFromExtractor` and 2 is `PedestalFromExtractorRndm`)

Dead and hot pixels are used in `magic-cta-pipe` for the MAGIC cleaning.

##### Simulation Configuration Data
Some general information about the simulated data, useful for IRF calculation, are read from the root files and stored in data.simulation container.

#### Changelog

-   v0.1: Initial version
-   v0.2.0: Unification of data and MC reading
-   v0.2.1: Monitoring data (Dead pixel and pedestal information)
-   v0.2.2: Added MC Header info
-   v0.2.3: Solve issue when interpolating information from drive reports, causing crashes when using pointing information in astropy SkyCoord objects. Make the reader faster when searching for ids of mono and stereo events
-   v0.2.4: fixes in mono_event_generator; fix to allow the use of relative paths as `input_url`
-   v0.3.0: update uproot to v4, since v3 is deprecated
-   v0.4.0: version compatible with ctapipe v0.12
-   v0.4.1: added CI, refactoring of code, added tests, extract drive information once
-   v0.4.2: added more tests, refactored code, allow the processing of all subruns from the same run at the same time (including drive information), correct de-rotation of quantities from the CORSIKA frame to the geographical frame, computation of bad pixels, modification of focal length to take into account the coma aberration, fix dowload of test data set
-   v0.4.3: difference of arrival times between events read from ROOT files, used for effective observation time calculation
-   v0.4.4: changed units of peak_time from time slices (as stored in MARS) to nanoseconds
-   v0.4.5: fixed automatic tests, add possibility to choose between effective and nominal focal length
