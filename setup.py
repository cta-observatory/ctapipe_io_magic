from setuptools import setup, find_packages
import os

tests_require = ["pytest"]

docs_require = [
    "sphinx~=4.2",
    "sphinx-automodapi",
    "sphinx_argparse",
    "sphinx_rtd_theme",
    "numpydoc",
    "nbsphinx",
]

setup(
    use_scm_version={"write_to": os.path.join("ctapipe_io_magic", "_version.py")},
    packages=find_packages(),
    install_requires=[
        "ctapipe~=0.19",
        "astropy~=5.0",
        "uproot~=5.0",
        "numpy>=1.20",
    ],
    package_data={
        "ctapipe_io_magic": ["resources/*"],
    },
    extras_require={
        "all": tests_require + docs_require,
        "tests": tests_require,
        "docs": docs_require,
    },
    setup_requires=["pytest_runner"],
)
