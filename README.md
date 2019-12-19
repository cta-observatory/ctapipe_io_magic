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

with MAGICEventSource(input_url=file_name) as source:
    for event in source:
        ...some processing...
```

The reader also works with multiple files parsed as wildcards, e.g.,

```python
MAGICEventSource(input_url=data_dir/*.root)
```

This is necessary to load and match stereo events, which are automatically created if data files from M1 and M2 for the same run are loaded. 

The reader is able to handle data or Monte Carlo files, which are automatically recognized. Note that the file names have to follow the convention:
- `*_M[1-2]_RUNNUMBER.SUBRUNNR_Y_*.root` for data
- `*_M[1-2]_za??to??_?_RUNNUMBER_Y_*.root` for Monte Carlos.

Note that currently, when loading multiple runs at once, the event ID is not unique.


#### Changelog

- v0.1: Initial version
- v0.2.0: Unification of data and MC reading
