# this is adapted from https://github.com/astropy/astropy/blob/master/astropy/version.py
# see https://github.com/astropy/astropy/pull/10774 for a discussion on why this needed.

try:
    try:
        from ._dev_version import version
    except ImportError:
        from ._version import version
except Exception:
    import warnings

    warnings.warn(
        "Could not determine ctapipe_io_magic version; this indicates a broken installation."
        " Install ctapipe_io_magic from PyPI, a local git repository or using "
        " pip install https://github.com/cta-observatory/ctapipe_io_magic@VERSION "
        " Installing github's autogenerated source release tarballs "
        " does not include version information and should be avoided."
    )
    del warnings
    version = "0.0.0"

__version__ = version
