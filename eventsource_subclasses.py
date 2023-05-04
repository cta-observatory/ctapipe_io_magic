from ctapipe.io import EventSource

for cls in EventSource.non_abstract_subclasses().values():
    print(cls.__name__)
