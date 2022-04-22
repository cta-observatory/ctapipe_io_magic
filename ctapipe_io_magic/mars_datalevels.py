from enum import Enum, auto


class OrderedEnum(Enum):
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class MARSDataLevel(OrderedEnum):
    """
    Enum of the different MARS Data Levels
    """

    CALIBRATED = auto()  # Calibrated images in charge and time (no waveforms)
    STAR = auto()  # Cleaned images, with Hillas parametrization
    SUPERSTAR = auto()  # Stereo parameters reconstructed
    MELIBEA = auto()  # Reconstruction of hadronness, event direction and energy
