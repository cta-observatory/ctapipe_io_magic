from setuptools import setup, find_packages
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='ctapipe_io_magic',
    packages=find_packages(),
    version='0.1',
    description='ctapipe plugin for reading of the calibrated MAGIC files',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'ctapipe',
        'astropy',
        'uproot',
        'numpy',
        'scipy'
    ],
    tests_require=['pytest'],
    setup_requires=['pytest_runner'],
    author='Ievgen Vovk',
    author_email='Ievgen.Vovk@mpp.mpg.de',
    license='MIT',
) 
