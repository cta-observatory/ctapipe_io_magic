from setuptools import setup, find_packages
import os

tests_require = ["pytest"]

docs_require = [
    "sphinx~=4.2",
    "sphinx-automodapi",
    "sphinx_argparse",
    "sphinx_rtd_theme",
    "numpydoc",
    "nbsphinx"
]

setup(
    use_scm_version={"write_to": os.path.join("ctapipe_io_magic", "_version.py")},
    packages=find_packages(),
    install_requires=[
        'ctapipe~=0.12.0',
        'astropy~=4.2',
        'uproot~=4.0',
        'numpy',
        'scipy'
    ],
    extras_require={
        "all": tests_require + docs_require,
        "tests": tests_require,
        "docs": docs_require,
    },
    setup_requires=['pytest_runner'],
)
