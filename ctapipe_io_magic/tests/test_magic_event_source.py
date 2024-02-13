import copy
import os
from pathlib import Path

import pytest

test_data = Path(os.getenv("MAGIC_TEST_DATA", "test_data")).absolute()
test_calibrated_real_dir = test_data / "real/calibrated"
test_calibrated_real = [
    test_calibrated_real_dir / "20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root",
    test_calibrated_real_dir / "20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root",
    test_calibrated_real_dir / "20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root",
    test_calibrated_real_dir / "20210314_M2_05095172.002_Y_CrabNebula-W0.40+035.root",
]

test_calibrated_real_only_events = [
    test_calibrated_real_dir
    / "missing_trees/20210314_M1_05095172.001_Y_CrabNebula-W0.40+035_only_events.root",
]

test_calibrated_real_only_drive = [
    test_calibrated_real_dir
    / "missing_trees/20210314_M1_05095172.001_Y_CrabNebula-W0.40+035_only_drive.root",
]

test_calibrated_real_only_runh = [
    test_calibrated_real_dir
    / "missing_trees/20210314_M1_05095172.001_Y_CrabNebula-W0.40+035_only_runh.root",
]

test_calibrated_real_only_trigger = [
    test_calibrated_real_dir
    / "missing_trees/20210314_M1_05095172.001_Y_CrabNebula-W0.40+035_only_trigger.root",
]

test_calibrated_real_without_prescaler_trigger = [
    test_calibrated_real_dir
    / "missing_prescaler_trigger/20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root",
    test_calibrated_real_dir
    / "missing_prescaler_trigger/20210314_M1_05095172.002_Y_CrabNebula-W0.40+035_no_prescaler_trigger.root",
]

test_calibrated_real_hast = [
    test_calibrated_real_dir / "20230324_M1_05106879.001_Y_1ES0806+524-W0.40+000.root",
    test_calibrated_real_dir / "20230324_M1_05106879.002_Y_1ES0806+524-W0.40+000.root",
    test_calibrated_real_dir / "20230324_M2_05106879.001_Y_1ES0806+524-W0.40+000.root",
    test_calibrated_real_dir / "20230324_M2_05106879.002_Y_1ES0806+524-W0.40+000.root",
]

test_calibrated_simulated_dir = test_data / "simulated/calibrated"
test_calibrated_simulated = [
    test_calibrated_simulated_dir / "GA_M1_za35to50_8_824318_Y_w0.root",
    test_calibrated_simulated_dir / "GA_M1_za35to50_8_824319_Y_w0.root",
    test_calibrated_simulated_dir / "GA_M2_za35to50_8_824318_Y_w0.root",
    test_calibrated_simulated_dir / "GA_M2_za35to50_8_824319_Y_w0.root",
]

test_calibrated_all = (
    test_calibrated_real + test_calibrated_simulated + test_calibrated_real_hast
)

test_calibrated_missing_trees = (
    test_calibrated_real_only_events
    + test_calibrated_real_only_drive
    + test_calibrated_real_only_runh
    + test_calibrated_real_only_trigger
)

data_dict = dict()

data_dict["20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root"] = dict()
data_dict["20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root"] = dict()
data_dict["20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root"] = dict()
data_dict["20210314_M2_05095172.002_Y_CrabNebula-W0.40+035.root"] = dict()
data_dict["20230324_M1_05106879.001_Y_1ES0806+524-W0.40+000.root"] = dict()
data_dict["20230324_M1_05106879.002_Y_1ES0806+524-W0.40+000.root"] = dict()
data_dict["20230324_M2_05106879.001_Y_1ES0806+524-W0.40+000.root"] = dict()
data_dict["20230324_M2_05106879.002_Y_1ES0806+524-W0.40+000.root"] = dict()
data_dict["GA_M1_za35to50_8_824318_Y_w0.root"] = dict()
data_dict["GA_M1_za35to50_8_824319_Y_w0.root"] = dict()
data_dict["GA_M2_za35to50_8_824318_Y_w0.root"] = dict()
data_dict["GA_M2_za35to50_8_824319_Y_w0.root"] = dict()

data_dict["20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root"]["n_events_tot"] = 500
data_dict["20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root"][
    "n_events_stereo"
] = 458
data_dict["20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root"][
    "n_events_pedestal"
] = 42
data_dict["20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root"][
    "n_events_mc_mono"
] = 0

data_dict["20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root"]["n_events_tot"] = 500
data_dict["20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root"][
    "n_events_stereo"
] = 452
data_dict["20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root"][
    "n_events_pedestal"
] = 48
data_dict["20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root"][
    "n_events_mc_mono"
] = 0

data_dict["20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root"]["n_events_tot"] = 500
data_dict["20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root"][
    "n_events_stereo"
] = 459
data_dict["20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root"][
    "n_events_pedestal"
] = 41
data_dict["20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root"][
    "n_events_mc_mono"
] = 0

data_dict["20210314_M2_05095172.002_Y_CrabNebula-W0.40+035.root"]["n_events_tot"] = 500
data_dict["20210314_M2_05095172.002_Y_CrabNebula-W0.40+035.root"][
    "n_events_stereo"
] = 450
data_dict["20210314_M2_05095172.002_Y_CrabNebula-W0.40+035.root"][
    "n_events_pedestal"
] = 50
data_dict["20210314_M2_05095172.002_Y_CrabNebula-W0.40+035.root"][
    "n_events_mc_mono"
] = 0

data_dict["20230324_M1_05106879.001_Y_1ES0806+524-W0.40+000.root"][
    "n_events_tot"
] = 1000
data_dict["20230324_M1_05106879.001_Y_1ES0806+524-W0.40+000.root"][
    "n_events_stereo"
] = 855
data_dict["20230324_M1_05106879.001_Y_1ES0806+524-W0.40+000.root"][
    "n_events_3_tel"
] = 477
data_dict["20230324_M1_05106879.001_Y_1ES0806+524-W0.40+000.root"][
    "n_events_2_tel_m1_lst"
] = 37
data_dict["20230324_M1_05106879.001_Y_1ES0806+524-W0.40+000.root"][
    "n_events_2_tel_m2_lst"
] = 0
data_dict["20230324_M1_05106879.001_Y_1ES0806+524-W0.40+000.root"][
    "n_events_2_tel_m1_m2"
] = 341
data_dict["20230324_M1_05106879.001_Y_1ES0806+524-W0.40+000.root"][
    "n_events_pedestal"
] = 142
data_dict["20230324_M1_05106879.001_Y_1ES0806+524-W0.40+000.root"][
    "n_events_mc_mono"
] = 0

data_dict["20230324_M1_05106879.002_Y_1ES0806+524-W0.40+000.root"][
    "n_events_tot"
] = 1000
data_dict["20230324_M1_05106879.002_Y_1ES0806+524-W0.40+000.root"][
    "n_events_stereo"
] = 853
data_dict["20230324_M1_05106879.002_Y_1ES0806+524-W0.40+000.root"][
    "n_events_3_tel"
] = 494
data_dict["20230324_M1_05106879.002_Y_1ES0806+524-W0.40+000.root"][
    "n_events_2_tel_m1_lst"
] = 34
data_dict["20230324_M1_05106879.002_Y_1ES0806+524-W0.40+000.root"][
    "n_events_2_tel_m2_lst"
] = 0
data_dict["20230324_M1_05106879.002_Y_1ES0806+524-W0.40+000.root"][
    "n_events_2_tel_m1_m2"
] = 325
data_dict["20230324_M1_05106879.002_Y_1ES0806+524-W0.40+000.root"][
    "n_events_pedestal"
] = 145
data_dict["20230324_M1_05106879.002_Y_1ES0806+524-W0.40+000.root"][
    "n_events_mc_mono"
] = 0

data_dict["20230324_M2_05106879.001_Y_1ES0806+524-W0.40+000.root"][
    "n_events_tot"
] = 1000
data_dict["20230324_M2_05106879.001_Y_1ES0806+524-W0.40+000.root"][
    "n_events_stereo"
] = 943
data_dict["20230324_M2_05106879.001_Y_1ES0806+524-W0.40+000.root"][
    "n_events_3_tel"
] = 226
data_dict["20230324_M2_05106879.001_Y_1ES0806+524-W0.40+000.root"][
    "n_events_2_tel_m1_lst"
] = 0
data_dict["20230324_M2_05106879.001_Y_1ES0806+524-W0.40+000.root"][
    "n_events_2_tel_m2_lst"
] = 642
data_dict["20230324_M2_05106879.001_Y_1ES0806+524-W0.40+000.root"][
    "n_events_2_tel_m1_m2"
] = 75
data_dict["20230324_M2_05106879.001_Y_1ES0806+524-W0.40+000.root"][
    "n_events_pedestal"
] = 57
data_dict["20230324_M2_05106879.001_Y_1ES0806+524-W0.40+000.root"][
    "n_events_mc_mono"
] = 0

data_dict["20230324_M2_05106879.002_Y_1ES0806+524-W0.40+000.root"][
    "n_events_tot"
] = 1000
data_dict["20230324_M2_05106879.002_Y_1ES0806+524-W0.40+000.root"][
    "n_events_stereo"
] = 949
data_dict["20230324_M2_05106879.002_Y_1ES0806+524-W0.40+000.root"][
    "n_events_3_tel"
] = 215
data_dict["20230324_M2_05106879.002_Y_1ES0806+524-W0.40+000.root"][
    "n_events_2_tel_m1_lst"
] = 0
data_dict["20230324_M2_05106879.002_Y_1ES0806+524-W0.40+000.root"][
    "n_events_2_tel_m2_lst"
] = 644
data_dict["20230324_M2_05106879.002_Y_1ES0806+524-W0.40+000.root"][
    "n_events_2_tel_m1_m2"
] = 90
data_dict["20230324_M2_05106879.002_Y_1ES0806+524-W0.40+000.root"][
    "n_events_pedestal"
] = 51
data_dict["20230324_M2_05106879.002_Y_1ES0806+524-W0.40+000.root"][
    "n_events_mc_mono"
] = 0

data_dict["GA_M1_za35to50_8_824318_Y_w0.root"]["n_events_tot"] = 99
data_dict["GA_M1_za35to50_8_824318_Y_w0.root"]["n_events_stereo"] = 67
data_dict["GA_M1_za35to50_8_824318_Y_w0.root"]["n_events_pedestal"] = 0
data_dict["GA_M1_za35to50_8_824318_Y_w0.root"]["n_events_mc_mono"] = 32

data_dict["GA_M1_za35to50_8_824319_Y_w0.root"]["n_events_tot"] = 111
data_dict["GA_M1_za35to50_8_824319_Y_w0.root"]["n_events_stereo"] = 80
data_dict["GA_M1_za35to50_8_824319_Y_w0.root"]["n_events_pedestal"] = 0
data_dict["GA_M1_za35to50_8_824319_Y_w0.root"]["n_events_mc_mono"] = 31

data_dict["GA_M2_za35to50_8_824318_Y_w0.root"]["n_events_tot"] = 118
data_dict["GA_M2_za35to50_8_824318_Y_w0.root"]["n_events_stereo"] = 67
data_dict["GA_M2_za35to50_8_824318_Y_w0.root"]["n_events_pedestal"] = 0
data_dict["GA_M2_za35to50_8_824318_Y_w0.root"]["n_events_mc_mono"] = 51

data_dict["GA_M2_za35to50_8_824319_Y_w0.root"]["n_events_tot"] = 132
data_dict["GA_M2_za35to50_8_824319_Y_w0.root"]["n_events_stereo"] = 80
data_dict["GA_M2_za35to50_8_824319_Y_w0.root"]["n_events_pedestal"] = 0
data_dict["GA_M2_za35to50_8_824319_Y_w0.root"]["n_events_mc_mono"] = 52


@pytest.mark.parametrize("dataset", test_calibrated_all)
def test_event_source_for_magic_file(dataset):
    from ctapipe.io import EventSource

    reader = EventSource(dataset)

    # import here to see if ctapipe detects plugin
    from ctapipe_io_magic import MAGICEventSource

    assert isinstance(reader, MAGICEventSource)
    assert reader.input_url == dataset


@pytest.mark.parametrize("dataset", test_calibrated_all)
def test_compatible(dataset):
    from ctapipe_io_magic import MAGICEventSource

    assert MAGICEventSource.is_compatible(dataset)


def test_not_compatible():
    from ctapipe_io_magic import MAGICEventSource

    assert MAGICEventSource.is_compatible(None) is False


@pytest.mark.parametrize("dataset", test_calibrated_all)
def test_stream(dataset):
    from ctapipe_io_magic import MAGICEventSource

    with MAGICEventSource(input_url=dataset, process_run=False) as source:
        assert not source.is_stream


def test_allowed_tels():
    from ctapipe_io_magic import MAGICEventSource
    import numpy as np

    dataset = (
        test_calibrated_real_dir
        / "20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root"
    )
    allowed_tels = {1}
    with MAGICEventSource(
        input_url=dataset, process_run=False, allowed_tels=allowed_tels
    ) as source:
        assert not allowed_tels.symmetric_difference(source.subarray.tel_ids)
        for event in source:
            assert set(event.trigger.tels_with_trigger).issubset(allowed_tels)
            assert set(event.pointing.tel).issubset(allowed_tels)


@pytest.mark.parametrize("dataset", test_calibrated_all)
def test_loop(dataset):
    from ctapipe_io_magic import MAGICEventSource

    n_events = 10
    with MAGICEventSource(
        input_url=dataset, max_events=n_events, process_run=False
    ) as source:
        for i, event in enumerate(source):
            assert event.count == i
            if "_M1_" in dataset.name:
                assert 1 in event.trigger.tels_with_trigger
                if not source.is_hast:
                    assert event.trigger.tels_with_trigger == [1, 2]
            if "_M2_" in dataset.name:
                assert 2 in event.trigger.tels_with_trigger
                if not source.is_hast:
                    assert event.trigger.tels_with_trigger == [1, 2]

        assert (i + 1) == n_events

        for event in source:
            # Check generator has restarted from beginning
            assert event.count == 0
            break


@pytest.mark.parametrize("dataset", test_calibrated_real)
def test_loop_pedestal(dataset):
    from ctapipe_io_magic import MAGICEventSource
    from ctapipe.containers import EventType

    n_events = 10
    with MAGICEventSource(
        input_url=dataset, max_events=n_events, use_pedestals=True, process_run=False
    ) as source:
        for event in source:
            assert event.trigger.event_type == EventType.SKY_PEDESTAL


@pytest.mark.parametrize("dataset", test_calibrated_all)
def test_number_of_events(dataset):
    from ctapipe_io_magic import MAGICEventSource

    with MAGICEventSource(input_url=dataset, process_run=False) as source:
        run = source._set_active_run(source.files_[0])
        assert (
            run["data"].n_cosmic_events
            == data_dict[source.input_url.name]["n_events_stereo"]
        )
        assert (
            run["data"].n_pedestal_events
            == data_dict[source.input_url.name]["n_events_pedestal"]
        )

        if source.is_hast:
            count_3_tel = 0
            count_2_tel_m1_lst = 0
            count_2_tel_m2_lst = 0
            count_2_tel_m1_m2 = 0
            for event in source:
                if event.trigger.tels_with_trigger == [1, 2, 3]:
                    count_3_tel += 1
                elif event.trigger.tels_with_trigger == [1, 3]:
                    count_2_tel_m1_lst += 1
                elif event.trigger.tels_with_trigger == [2, 3]:
                    count_2_tel_m2_lst += 1
                elif event.trigger.tels_with_trigger == [1, 2]:
                    count_2_tel_m1_m2 += 1

            assert count_3_tel == data_dict[source.input_url.name]["n_events_3_tel"]
            assert (
                count_2_tel_m1_lst
                == data_dict[source.input_url.name]["n_events_2_tel_m1_lst"]
            )
            assert (
                count_2_tel_m2_lst
                == data_dict[source.input_url.name]["n_events_2_tel_m2_lst"]
            )
            assert (
                count_2_tel_m1_m2
                == data_dict[source.input_url.name]["n_events_2_tel_m1_m2"]
            )

        # if '_M1_' in dataset.name:
        #     assert run['data'].n_cosmics_stereo_events_m1 == data_dict[source.input_url.name]['n_events_stereo']
        #     assert run['data'].n_pedestal_events_m1 == data_dict[source.input_url.name]['n_events_pedestal']
        # if '_M2_' in dataset.name:
        #     assert run['data'].n_cosmics_stereo_events_m2 == data_dict[source.input_url.name]['n_events_stereo']
        #     assert run['data'].n_pedestal_events_m2 == data_dict[source.input_url.name]['n_events_pedestal']


@pytest.mark.parametrize("dataset", test_calibrated_all)
def test_run_info(dataset):
    from ctapipe_io_magic import MAGICEventSource

    with MAGICEventSource(input_url=dataset, process_run=False) as source:
        run_info = [
            MAGICEventSource.get_run_info_from_name(item.name)
            for item in source.file_list
        ]
        run_numbers = [i[0] for i in run_info]
        is_mc = [i[1] for i in run_info][0]
        telescope = [i[2] for i in run_info][0]
        datalevel = [i[3] for i in run_info][0]
        assert run_numbers == [source.run_id]
        assert is_mc == source.is_simulation
        assert telescope == source.telescope
        assert datalevel == source.mars_datalevel
        assert source.is_stereo == True
        assert source.is_sumt == False
        if "1ES0806" in dataset.name:
            assert source.is_hast == True


def test_multiple_runs_real():
    from ctapipe_io_magic import MAGICEventSource
    from ctapipe.containers import EventType

    real_data_mask = (
        test_calibrated_real_dir
        / "20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root"
    )

    n_events = 600
    with MAGICEventSource(input_url=real_data_mask, max_events=n_events) as source:
        for i, event in enumerate(source):
            assert event.trigger.event_type == EventType.SUBARRAY
            assert event.count == i
            assert source.telescope in event.trigger.tels_with_trigger
            assert event.trigger.tels_with_trigger == [1, 2]

        assert (i + 1) == n_events

        for event in source:
            # Check generator has restarted from beginning
            assert event.count == 0
            break


def test_subarray_multiple_runs():
    from ctapipe_io_magic import MAGICEventSource

    simulated_data_mask = (
        test_calibrated_simulated_dir / "GA_M1_za35to50_8_824318_Y_w0.root"
    )

    source = MAGICEventSource(input_url=simulated_data_mask)
    sim_config = source.simulation_config
    assert list(sim_config.keys()) == source.obs_ids


@pytest.mark.parametrize("dataset", test_calibrated_all)
def test_that_event_is_not_modified_after_loop(dataset):
    from ctapipe_io_magic import MAGICEventSource

    n_events = 10
    with MAGICEventSource(
        input_url=dataset, max_events=n_events, process_run=False
    ) as source:
        for event in source:
            last_event = copy.deepcopy(event)

        # now `event` should be identical with the deepcopy of itself from
        # inside the loop.
        # Unfortunately this does not work:
        #      assert last_event == event
        # So for the moment we just compare event ids
        assert event.index.event_id == last_event.index.event_id


@pytest.mark.parametrize("dataset", test_calibrated_all)
def test_geom(dataset):
    from ctapipe_io_magic import MAGICEventSource

    with MAGICEventSource(input_url=dataset) as source:
        assert source.subarray.tels[1].camera.geometry.pix_x.size == 1039
        assert source.subarray.tels[2].camera.geometry.pix_x.size == 1039


@pytest.mark.parametrize("dataset", test_calibrated_all)
def test_focal_length_choice(dataset):
    from astropy import units as u
    from ctapipe_io_magic import MAGICEventSource
    from ctapipe.instrument import FocalLengthKind

    with MAGICEventSource(
        input_url=dataset,
        process_run=False,
        focal_length_choice=FocalLengthKind.EQUIVALENT,
    ) as source:
        assert source.subarray.tel[1].optics.equivalent_focal_length == u.Quantity(
            16.97, u.m
        )
        assert source.subarray.tel[2].optics.equivalent_focal_length == u.Quantity(
            16.97, u.m
        )
        assert source.subarray.tel[1].camera.geometry.frame.focal_length == u.Quantity(
            16.97, u.m
        )
        assert source.subarray.tel[2].camera.geometry.frame.focal_length == u.Quantity(
            16.97, u.m
        )

    with MAGICEventSource(
        input_url=dataset,
        process_run=False,
        focal_length_choice=FocalLengthKind.EFFECTIVE,
    ) as source:
        assert source.subarray.tel[1].optics.effective_focal_length == u.Quantity(
            17 * 1.0713, u.m
        )
        assert source.subarray.tel[2].optics.effective_focal_length == u.Quantity(
            17 * 1.0713, u.m
        )
        assert source.subarray.tel[1].camera.geometry.frame.focal_length == u.Quantity(
            17 * 1.0713, u.m
        )
        assert source.subarray.tel[2].camera.geometry.frame.focal_length == u.Quantity(
            17 * 1.0713, u.m
        )


@pytest.mark.parametrize("dataset", test_calibrated_missing_trees)
def test_check_files(dataset):
    from ctapipe_io_magic import MAGICEventSource, FailedFileCheckError

    with pytest.raises(FailedFileCheckError):
        MAGICEventSource(input_url=dataset, process_run=False)


def test_check_missing_files():
    from ctapipe_io_magic import MAGICEventSource, MissingInputFilesError

    with pytest.raises(MissingInputFilesError):
        MAGICEventSource(
            input_url="20501312_M1_05095172.001_Y_FakeSource-W0.40+035.root",
            process_run=False,
        )


def test_broken_subruns_missing_trees():
    from ctapipe_io_magic import MAGICEventSource

    input_file = test_calibrated_real_dir / "missing_prescaler_trigger/20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root"

    MAGICEventSource(input_url=input_file, process_run=True,)


def test_broken_subruns_missing_arrays():
    from ctapipe_io_magic import MAGICEventSource

    input_file = (
        test_calibrated_real_dir
        / "missing_arrays/20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root"
    )

    MAGICEventSource(
        input_url=input_file,
        process_run=True,
    )
