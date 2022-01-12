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

test_calibrated_all = test_calibrated_real+test_calibrated_simulated

data_dict = dict()

data_dict['20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root'] = dict()
data_dict['20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root'] = dict()
data_dict['20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root'] = dict()
data_dict['20210314_M2_05095172.002_Y_CrabNebula-W0.40+035.root'] = dict()
data_dict['GA_M1_za35to50_8_824318_Y_w0.root'] = dict()
data_dict['GA_M1_za35to50_8_824319_Y_w0.root'] = dict()
data_dict['GA_M2_za35to50_8_824318_Y_w0.root'] = dict()
data_dict['GA_M2_za35to50_8_824319_Y_w0.root'] = dict()

data_dict['20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root']['n_events_tot'] = 500
data_dict['20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root']['n_events_stereo'] = 458
data_dict['20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root']['n_events_pedestal'] = 42
data_dict['20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root']['n_events_mc_mono'] = 0

data_dict['20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root']['n_events_tot'] = 500
data_dict['20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root']['n_events_stereo'] = 452
data_dict['20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root']['n_events_pedestal'] = 48
data_dict['20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root']['n_events_mc_mono'] = 0

data_dict['20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root']['n_events_tot'] = 500
data_dict['20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root']['n_events_stereo'] = 459
data_dict['20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root']['n_events_pedestal'] = 41
data_dict['20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root']['n_events_mc_mono'] = 0

data_dict['20210314_M2_05095172.002_Y_CrabNebula-W0.40+035.root']['n_events_tot'] = 500
data_dict['20210314_M2_05095172.002_Y_CrabNebula-W0.40+035.root']['n_events_stereo'] = 450
data_dict['20210314_M2_05095172.002_Y_CrabNebula-W0.40+035.root']['n_events_pedestal'] = 50
data_dict['20210314_M2_05095172.002_Y_CrabNebula-W0.40+035.root']['n_events_mc_mono'] = 0

data_dict['GA_M1_za35to50_8_824318_Y_w0.root']['n_events_tot'] = 99
data_dict['GA_M1_za35to50_8_824318_Y_w0.root']['n_events_stereo'] = 67
data_dict['GA_M1_za35to50_8_824318_Y_w0.root']['n_events_pedestal'] = 0
data_dict['GA_M1_za35to50_8_824318_Y_w0.root']['n_events_mc_mono'] = 32

data_dict['GA_M1_za35to50_8_824319_Y_w0.root']['n_events_tot'] = 111
data_dict['GA_M1_za35to50_8_824319_Y_w0.root']['n_events_stereo'] = 80
data_dict['GA_M1_za35to50_8_824319_Y_w0.root']['n_events_pedestal'] = 0
data_dict['GA_M1_za35to50_8_824319_Y_w0.root']['n_events_mc_mono'] = 31

data_dict['GA_M2_za35to50_8_824318_Y_w0.root']['n_events_tot'] = 118
data_dict['GA_M2_za35to50_8_824318_Y_w0.root']['n_events_stereo'] = 67
data_dict['GA_M2_za35to50_8_824318_Y_w0.root']['n_events_pedestal'] = 0
data_dict['GA_M2_za35to50_8_824318_Y_w0.root']['n_events_mc_mono'] = 51

data_dict['GA_M2_za35to50_8_824319_Y_w0.root']['n_events_tot'] = 132
data_dict['GA_M2_za35to50_8_824319_Y_w0.root']['n_events_stereo'] = 80
data_dict['GA_M2_za35to50_8_824319_Y_w0.root']['n_events_pedestal'] = 0
data_dict['GA_M2_za35to50_8_824319_Y_w0.root']['n_events_mc_mono'] = 52


@pytest.mark.parametrize('dataset', test_calibrated_all)
def test_event_source_for_magic_file(dataset):
    from ctapipe.io import EventSource

    reader = EventSource(dataset)

    # import here to see if ctapipe detects plugin
    from ctapipe_io_magic import MAGICEventSource

    assert isinstance(reader, MAGICEventSource)
    assert reader.input_url == dataset


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
            if "_M1_" in dataset.name:
                assert event.trigger.tels_with_trigger == [1]
            if "_M2_" in dataset.name:
                assert event.trigger.tels_with_trigger == [2]

        assert (i + 1) == n_events

        for event in source:
            # Check generator has restarted from beginning
            assert event.count == 0
            break


@pytest.mark.parametrize('dataset', test_calibrated_real)
def test_loop_pedestal(dataset):
    from ctapipe_io_magic import MAGICEventSource
    from ctapipe_io_magic.constants import PEDESTAL_TRIGGER_PATTERN
    n_events = 10
    with MAGICEventSource(input_url=dataset, max_events=n_events, use_pedestals=True) as source:
        for event in source:
            assert event.trigger.event_type == PEDESTAL_TRIGGER_PATTERN


@pytest.mark.parametrize('dataset', test_calibrated_all)
def test_number_of_events(dataset):
    from ctapipe_io_magic import MAGICEventSource

    with MAGICEventSource(input_url=dataset) as source:
        run = source._set_active_run(run_number=source.run_numbers)
        if '_M1_' in dataset.name:
            assert run['data'].n_mono_events_m1 == data_dict[source.input_url.name]['n_events_stereo']
            assert run['data'].n_pedestal_events_m1 == data_dict[source.input_url.name]['n_events_pedestal']
        if '_M2_' in dataset.name:
            assert run['data'].n_mono_events_m2 == data_dict[source.input_url.name]['n_events_stereo']
            assert run['data'].n_pedestal_events_m2 == data_dict[source.input_url.name]['n_events_pedestal']


@pytest.mark.parametrize('dataset', test_calibrated_all)
def test_run_info(dataset):
    from ctapipe_io_magic import MAGICEventSource

    with MAGICEventSource(input_url=dataset) as source:
        run_info = MAGICEventSource.get_run_info_from_name(str(source.input_url))
        run_number = run_info[0]
        is_mc = run_info[1]
        telescope = run_info[2]
        datalevel = run_info[3]
        assert run_number == source.run_numbers
        assert run_number == source.obs_ids[0]
        assert is_mc == source.is_simulation
        assert telescope == source.telescope
        assert datalevel == source.mars_datalevel


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
