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

