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
