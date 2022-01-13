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

With more recent versions of *ctapipe*, only one file at a time can be read. This means that in the case of MAGIC calibrated files,
data is loaded subrun by subrun. No matching of the events is performed at this level (if stereo data).

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
event_generator = MAGICEventSource(input_url=file_name)
for cosmic_event in event_generator:
    ...some processing...

pedestal_event_generator = MAGICEventSource(input_url=file_name, use_pedestals=True)
for pedestal_event in pedestal_event_generator:
    ...some processing...
```

#### Features

##### Monitoring data

Monitoring data are saved in `run['data'].monitoring_data` and can also accessed event-wise via the `event.mon` container. Even if they can be accessed event-wise, they are saved only once per run, i.e., identical for all events in a run. If monitoring data is taken several times during a run, the `run['data'].monitoring_data`/`event.mon` sub-containers contain arrays of the quantities taken at the different times together with an array of the time stamps. So far, we have:

-   Dead pixel information (MARS `RunHeaders.MBadPixelsCam.fArray.fInfo` tree), once per sub-run in `run['data'].monitoring_data['MX']['badpixelinfo']` (with X=1 or X=2) or `event.mon.tel[tel_id].pixel_status`
-   Pedestal information from MARS `Pedestals` tree to calculate hot pixels in `event.mon.tel[tel_id].pedestal` or:
    -   `run['data'].monitoring_data['MX']['PedestalFundamental']`
    -   `run['data'].monitoring_data['MX']['PedestalFromExtractor']`
    -   `run['data'].monitoring_data['MX']['PedestalFromExtractorRndm']`

Dead pixel and pedestal information are read by `magic-cta-pipe` `MAGIC_Badpixels.py` class.

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
