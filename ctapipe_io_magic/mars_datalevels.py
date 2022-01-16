from enum import Enum, auto


class MARSDataLevel(Enum):
    """
    Enum of the different MARS Data Levels
    """

    CALIBRATED = auto()  # Calibrated images in charge and time (no waveforms)
    STAR = auto()  # Cleaned images, with Hillas parametrization
    SUPERSTAR = auto()  # Stereo parameters reconstructed
    MELIBEA = auto()  # Reconstruction of hadronness, event direction and energy
