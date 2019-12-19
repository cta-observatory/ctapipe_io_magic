## *ctapipe* MAGIC event source

This module implements the *ctapipe* class, needed to read the calibrated data of the MAGIC telescope system. It requires the [*ctapipe*](https://github.com/cta-observatory/ctapipe) and [*uproot*](https://github.com/scikit-hep/uproot) packages to run.

#### Installation

Provided *ctapipe* is already installed, the installation can be done like so:

```bash
git clone https://gitlab.mpcdf.mpg.de/ievo/ctapipe_io_magic.git
pip install ./ctapipe_io_magic/
```

This installation via pip (provided, pip is installed) has the advantage to be nicely controlled for belonging to a given conda environment (and to be uninstalled). Alternatively, do

```bash
git clone https://gitlab.mpcdf.mpg.de/ievo/ctapipe_io_magic.git
cd ctapipe_io_magic
python setup.py install --user
```

#### Usage

```python
import ctapipe
from ctapipe_io_magic import MAGICEventSource

with MAGICEventSource(input_url=file_name) as event_source:
    for event in event_source:
        ...some processing...
```

The reader also works with multiple files parsed as wildcards, e.g.,

```python
event_source = MAGICEventSource(input_url='data_dir/*.root')
```

This is necessary to load and match stereo events, which are automatically created if data files from M1 and M2 for the same run are loaded. 

The reader is able to handle data or Monte Carlo files, which are automatically recognized. Note that the file names have to follow the convention:
- `*_M[1-2]_RUNNUMBER.SUBRUNNR_Y_*.root` for data
- `*_M[1-2]_za??to??_?_RUNNUMBER_Y_*.root` for Monte Carlos.

Note that currently, when loading multiple runs at once, the event ID is not unique.

##### More usage
Select a single run:
```python
run = event_source._set_active_run(event_source.run_numbers[0])
for n in range(run['data'].n_stereo_events):
    run['data'].get_stereo_event_data(n)
for n in range(run['data'].n_mono_events_m1):
    run['data'].get_mono_event_data(n, 'M1')
for n in range(run['data'].n_pedestal_events_m1):
    run['data'].get_pedestal_event_data(n, 'M1')
```

Select mono/pedestal events over event generator:
```python
mono_event_generator = event_source._mono_event_generator(telescope='M1')
for m1_mono_event in mono_event_generator:
     ...some processing...
pedestal_event_generator = event_source._pedestal_event_generator(telescope='M1')
...
```


#### Changelog

- v0.1: Initial version
- v0.2.0: Unification of data and MC reading
