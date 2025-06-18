import astropy.units as u
from astropy.coordinates import EarthLocation

# trigger patterns:
# mono events trigger pattern in stereo MC files
MC_MONO_TRIGGER_PATTERN = 0
DATA_MONO_TRIGGER_PATTERN = 1
# also for mono MC files (but stereo event id is 0 for all events)
MC_STEREO_AND_MONO_TRIGGER_PATTERN = 1
PEDESTAL_TRIGGER_PATTERN = 8
DATA_MONO_SUMT_TRIGGER_PATTERN = 32
MC_SUMT_TRIGGER_PATTERN = 32
# also for data taken in stereo with SumTrigger
DATA_STEREO_TRIGGER_PATTERN = 128
# additional trigger patterns for hardware stereo trigger
# topological trigger is one MAGIC and LST
DATA_TOPOLOGICAL_TRIGGER = 4
# stereo + topological trigger is M1+M2+LST
DATA_MAGIC_LST_TRIGGER = 132

#reference location from ctapipe_io_lst.constants
#: Area averaged position of LST-1, MAGIC-1 and MAGIC-2 (using 23**2 and 17**2 m2)
REFERENCE_LOCATION = EarthLocation(
    lon=-17.890879 * u.deg,
    lat=28.761579 * u.deg,
    height=2199 * u.m,  # MC obs-level
)
