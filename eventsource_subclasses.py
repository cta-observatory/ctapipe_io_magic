from ctapipe.io import EventSource
from ctapipe.core import non_abstract_children

for cls in non_abstract_children(EventSource):
    print(cls.__name__)
