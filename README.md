# ctapipe_io_magic [![Build Status](https://github.com/cta-observatory/ctapipe_io_magic/workflows/CI/badge.svg?branch=master)](https://github.com/cta-observatory/ctapipe_io_magic/actions?query=workflow%3ACI+branch%3Amaster) [![PyPI version](https://badge.fury.io/py/ctapipe-io-magic.svg)](https://badge.fury.io/py/ctapipe-io-magic) [![Conda version](https://anaconda.org/conda-forge/ctapipe-io-magic/badges/version.svg)](https://anaconda.org/conda-forge/ctapipe-io-magic)

## *ctapipe* MAGIC event source

EventSource plugin for *ctapipe*, needed to read the calibrated data of the MAGIC telescope system. It requires [*ctapipe*](https://github.com/cta-observatory/ctapipe) (v0.19.x) and [*uproot*](https://github.com/scikit-hep/uproot5) (>=5) packages to run.

#### Installation

Since version 0.5.4, `ctapipe_io_magic` is on conda-forge ([here](https://anaconda.org/conda-forge/ctapipe-io-magic)), which is the easiest way to install it.

To install into an exisiting environtment, just do:
```
# or conda
$ mamba install -c conda-forge ctapipe-io-magic
```

or, to create a new environment:
```
# or conda
mamba create -c conda-forge -n magicio python=3.11 ctapipe-io-magic
```

Alternatively, you can always clone the repository and install like in the following:

```bash
git clone https://github.com/cta-observatory/ctapipe_io_magic.git
cd ctapipe_io_magic
conda env create -n ctapipe-io_magic -f environment.yml
conda activate ctapipe-io_magic
pip install .
```

`ctapipe_io_magic` is also in PyPI (see [here](https://pypi.org/project/ctapipe-io-magic/)).

#### Test Data

To run the tests, a set of non-public files is needed. If you are a member of MAGIC, ask one of the project maintainers for the credentials and then run:

```bash
./download_test_data.sh
```

#### Usage

```python
import ctapipe
from ctapipe_io_magic import MAGICEventSource

with MAGICEventSource(input_url=file_name) as event_source:
    for event in event_source:
        ...some processing...
```

With more recent versions of *ctapipe*, only one file at a time can be read. However, by default if a subrun of calibrated data is given as input, `MAGICEventSource` will read the events from all the subruns from the run to which the data file belongs. To suppress this behavior, set `process_run=False` No matching of the events is performed at this level (if stereo data).

Starting from v0.4.7, `MAGICEventSource` will automatically recognize the type of data contained in the calibrated ROOT files (stereo or mono; std trigger or SumT). For MC data, in the case stereo MC data are to be used for mono analysis, one can set to True the `use_mc_mono_events` option of the `MAGICEventSource` to use also mono triggered events.

Pedestal events (trigger pattern = 8) can be generated as well.

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
-   v0.4.6: add support to read in data taken in mono mode (full for real data, partial for MCs). Fixed bug in recognition of mono/stereo or standard trigger/SumT data (added also for MC)
-   v0.4.7: add full support to read in real and MC data taken in mono mode, and with SumT. Added treatment of unsuitable pixels for MC data. Added readout of true XMax value from MC data (usually not available, filled with 0 otherwise)
-   v0.5.0: release compatible with ctapipe 0.17. Also, the equivalent focal length is set to the
    correct value used in MAGIC simulations (i.e. 16.97 meters)
-   v0.5.1: release compatible with ctapipe 0.19
-   v0.5.2: introduce capability of reading data taken with the Hardware Stereo Trigger (HaST) between MAGIC and LST-1. Also, fixed bug when getting the time difference between events for mono data
-   v0.5.3: support for python 3.11
-   v0.5.4: change license to BSD-3, add badges in README, do not use default channel in environment
-   v0.5.5: read LIDAR information, add checks for missing trees in the input files, take prescaler and trigger information from previous subruns if they are missing, add release-drafter and PyPI deploy workflows
-   v0.5.6: small maintenance changes
-   v0.5.7: changes for ctapipe v0.25
