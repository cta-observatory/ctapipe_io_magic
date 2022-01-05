def test_version():
    from ctapipe_io_magic import __version__

    assert __version__ != 'unknown'
