import copy
import pytest
from ctapipe_io_magic import MAGICEventSource
from ctapipe.io.eventseeker import EventSeeker
from ctapipe.utils import get_dataset_path


def test_compatible():
    m1_dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")
    m2_dataset = get_dataset_path("20131004_M2_05029747.003_Y_MagicCrab-W0.40+035.root")
    assert MAGICEventSource.is_compatible(m1_dataset)
    assert MAGICEventSource.is_compatible(m2_dataset)


def test_stream():
    dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")
    with MAGICEventSource(input_url=dataset) as source:
        assert not source.is_stream


def test_loop():
    dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")
    with MAGICEventSource(input_url=dataset) as source:
        count = 0
        for event in source:
            assert event.trigger.tels_with_trigger == [1]
            assert event.count == count
            count += 1

        for event in source:
            # Check generator has restarted from beginning
            assert event.count == 0
            break


def test_that_event_is_not_modified_after_loop():
    dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")

    # with MAGICEventSource(input_url=dataset, max_events=3) as source:
    with MAGICEventSource(input_url=dataset) as source:
        for event in source:
            last_event = copy.deepcopy(event)

        # now `event` should be identical with the deepcopy of itself from
        # inside the loop.
        # Unfortunately this does not work:
        #      assert last_event == event
        # So for the moment we just compare event ids
        assert event.index.event_id == last_event.index.event_id


def test_len():
    dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")

    with MAGICEventSource(input_url=dataset) as source:
        count = 0
        for _ in source:
            count += 1

        # assert count == len(source)
        n_stereo_events = source.current_run['data'].n_mono_events_m1
        assert count == n_stereo_events

    # with MAGICEventSource(input_url=dataset, max_events=3) as reader:
    # with MAGICEventSource(input_url=dataset) as reader:
    #     assert len(reader) == 3


def test_geom():
    dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")

    with MAGICEventSource(input_url=dataset) as source:
        event = next(source._generator())
        assert source.subarray.tels[1].camera.geometry.pix_x.size == 1039
        assert source.subarray.tels[2].camera.geometry.pix_x.size == 1039


def test_eventseeker():
    dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")

    with MAGICEventSource(input_url=dataset) as source:
        seeker = EventSeeker(source)
        event = seeker.get_event_index(0)
        assert event.count == 0
        assert event.index.event_id == 29795

        event = seeker.get_event_index(2)
        assert event.count == 2
        assert event.index.event_id == 29798

def test_eventcontent():
    dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")

    with MAGICEventSource(input_url=dataset) as source:
        seeker = EventSeeker(source)
        event = seeker.get_event_index(0)
        assert event.dl1.tel[1].image[0] == -0.53125
        assert event.dl1.tel[1].peak_time[0] == 49.125
