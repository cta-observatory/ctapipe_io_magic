## *ctapipe* MAGIC event source

This module implements the *ctapipe* class, needed to read the calibrated data of the MAGIC telescope system. It requires the [*ctapipe*](https://github.com/cta-observatory/ctapipe) and [*uproot*](https://github.com/scikit-hep/uproot) packages to run.

#### Installation

Provided *ctapipe* is already installed, the installation can be done like so:

```bash
https://github.com/cta-observatory/ctapipe_io_magic.git
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

