from ctapipe_io_magic import MARSDataLevel


def test_marsdatalevels():
    assert MARSDataLevel.CALIBRATED <= MARSDataLevel.STAR
    assert MARSDataLevel.SUPERSTAR >= MARSDataLevel.STAR
