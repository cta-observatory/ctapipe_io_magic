import copy
import os
from pathlib import Path

import pytest

test_data = Path(os.getenv('MAGIC_TEST_DATA', 'test_data')).absolute()
test_calibrated_real_dir = test_data / 'real/calibrated'
test_calibrated_real = [
    test_calibrated_real_dir / '20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root',
    test_calibrated_real_dir / '20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root',
    test_calibrated_real_dir / '20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root',
    test_calibrated_real_dir / '20210314_M2_05095172.002_Y_CrabNebula-W0.40+035.root',
]

test_calibrated_simulated_dir = test_data / 'simulated/calibrated'
test_calibrated_simulated = [
    test_calibrated_simulated_dir / 'GA_M1_za35to50_8_824318_Y_w0.root',
    test_calibrated_simulated_dir / 'GA_M1_za35to50_8_824319_Y_w0.root',
    test_calibrated_simulated_dir / 'GA_M2_za35to50_8_824318_Y_w0.root',
    test_calibrated_simulated_dir / 'GA_M2_za35to50_8_824319_Y_w0.root',
]

test_calibrated_all = [
    test_calibrated_real,
    test_calibrated_simulated,
]


@pytest.mark.parametrize('dataset', test_calibrated_all)
def test_compatible(dataset):
    from ctapipe_io_magic import MAGICEventSource
    assert MAGICEventSource.is_compatible(dataset)


@pytest.mark.parametrize('dataset', test_calibrated_all)
def test_stream(dataset):
    from ctapipe_io_magic import MAGICEventSource
    with MAGICEventSource(input_url=dataset) as source:
        assert not source.is_stream


@pytest.mark.parametrize('dataset', test_calibrated_all)
def test_loop(dataset):
    from ctapipe_io_magic import MAGICEventSource
    n_events = 10
    with MAGICEventSource(input_url=dataset, max_events=n_events) as source:
        for i, event in enumerate(source):
            assert event.count == i
            if "_M1_" in dataset:
                assert event.trigger.tels_with_trigger == [1]
            if "_M2_" in dataset:
                assert event.trigger.tels_with_trigger == [2]

        assert (i + 1) == n_events

        for event in source:
            # Check generator has restarted from beginning
            assert event.count == 0
            break


@pytest.mark.parametrize('dataset', test_calibrated_all)
def test_that_event_is_not_modified_after_loop(dataset):
    from ctapipe_io_magic import MAGICEventSource
    n_events = 10
    with MAGICEventSource(input_url=dataset, max_events=n_events) as source:
        for event in source:
            last_event = copy.deepcopy(event)

        # now `event` should be identical with the deepcopy of itself from
        # inside the loop.
        # Unfortunately this does not work:
        #      assert last_event == event
        # So for the moment we just compare event ids
        assert event.index.event_id == last_event.index.event_id


@pytest.mark.parametrize('dataset', test_calibrated_all)
def test_len(dataset):
    from ctapipe_io_magic import MAGICEventSource
    n_events = 10

    with MAGICEventSource(input_url=dataset, max_events=n_events) as source:
        count = 0
        for _ in source:
            count += 1

        # assert count == len(source)
        n_stereo_events = source.current_run['data'].n_mono_events_m1
        assert count == n_stereo_events

    # with MAGICEventSource(input_url=dataset, max_events=3) as reader:
    # with MAGICEventSource(input_url=dataset) as reader:
    #     assert len(reader) == 3


@pytest.mark.parametrize('dataset', test_calibrated_all)
def test_geom(dataset):
    from ctapipe_io_magic import MAGICEventSource

    with MAGICEventSource(input_url=dataset) as source:
        assert source.subarray.tels[1].camera.geometry.pix_x.size == 1039
        assert source.subarray.tels[2].camera.geometry.pix_x.size == 1039


# def test_eventseeker():
#    dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")
#
#    with MAGICEventSource(input_url=dataset) as source:
#        seeker = EventSeeker(source)
#        event = seeker.get_event_index(0)
#        assert event.count == 0
#        assert event.index.event_id == 29795
#
#        event = seeker.get_event_index(2)
#        assert event.count == 2
#        assert event.index.event_id == 29798
#
# def test_eventcontent():
#    dataset = get_dataset_path("20131004_M1_05029747.003_Y_MagicCrab-W0.40+035.root")
#
#    with MAGICEventSource(input_url=dataset) as source:
#        seeker = EventSeeker(source)
#        event = seeker.get_event_index(0)
#        assert event.dl1.tel[1].image[0] == -0.53125
#        assert event.dl1.tel[1].peak_time[0] == 49.125
