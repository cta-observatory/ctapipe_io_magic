from pathlib import Path
import os

from ctapipe.io import read_table
from ctapipe.containers import EventType
import numpy as np

test_data = Path(os.getenv("MAGIC_TEST_DATA", "test_data")).absolute()
test_cal_path = (
    test_data / "real/calibrated/20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root"
)
config = Path(".").absolute() / "example_stage1_config.json"


def test_stage1_multiple_runs():
    """Test the ctapipe process tool can read in MAGIC real data using the event source"""
    from ctapipe.tools.process import ProcessorTool
    from ctapipe.core import run_tool

    tool = ProcessorTool()
    output = (
        test_data / "real/calibrated/20210314_M1_05095172_Y_CrabNebula-W0.40+035.h5"
    )

    ret = run_tool(
        tool,
        argv=[
            f"--input={test_cal_path}",
            f"--output={output}",
            f"--config={str(config)}",
            "--camera-frame",
        ],
    )
    assert ret == 0

    parameters = read_table(output, "/dl1/event/telescope/parameters/tel_001")
    assert len(parameters) == 910

    trigger = read_table(output, "/dl1/event/subarray/trigger")

    event_type_counts = np.bincount(trigger["event_type"])

    # no pedestals expected, should be only physics data
    assert event_type_counts.sum() == 910
    assert event_type_counts[EventType.FLATFIELD.value] == 0
    assert event_type_counts[EventType.SKY_PEDESTAL.value] == 0
    assert event_type_counts[EventType.SUBARRAY.value] == 910


def test_stage1_single_run():
    """Test the ctapipe process tool can read in MAGIC real data using the event source"""
    from ctapipe.tools.process import ProcessorTool
    from ctapipe.core import run_tool

    tool = ProcessorTool()
    output = test_cal_path.with_suffix(".h5")

    ret = run_tool(
        tool,
        argv=[
            f"--input={test_cal_path}",
            f"--output={output}",
            f"--config={str(config)}",
            "--MAGICEventSource.process_run=false",
            "--camera-frame",
        ],
    )
    assert ret == 0

    parameters = read_table(output, "/dl1/event/telescope/parameters/tel_001")
    assert len(parameters) == 458

    trigger = read_table(output, "/dl1/event/subarray/trigger")

    event_type_counts = np.bincount(trigger["event_type"])

    # no pedestals expected, should be only physics data
    assert event_type_counts.sum() == 458
    assert event_type_counts[EventType.FLATFIELD.value] == 0
    assert event_type_counts[EventType.SKY_PEDESTAL.value] == 0
    assert event_type_counts[EventType.SUBARRAY.value] == 458
