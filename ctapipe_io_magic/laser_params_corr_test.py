import logging
import datetime
from pathlib import Path
import os
import numpy as np
import astropy.units as u
from ctapipe_io_magic import MAGICEventSource
import zd_az_corr
import params
from switch_times import switch_times

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

class ReportLaser:
    def magic_reports_reading(self, input_file, process_run=False):
        logger.info(f"\nInput file: {input_file}")
        event_source = MAGICEventSource(input_file, process_run=process_run)
        is_simulation = event_source.is_simulation
        logger.info(f"\nIs simulation: {is_simulation}")

        obs_id = event_source.obs_ids[0]
        tel_id = event_source.telescope

        logger.info(f"\nObservation ID: {obs_id}")
        logger.info(f"Telescope ID: {tel_id}")

        laser = event_source.laser

        for key, report_list in laser.items():
            for report in report_list:
                logger.info(f"Accessing parameters for each ReportLaserContainer object")

        return report

    def get_c0(self, time, zenith, range_max):
        C_0 = params.gkC_0Period1  # absolute calibration at beginning of 2013
        if params.stime21 > 0:
            fFullOverlap = params.gkFullOverlap5  # overlap has moved, after installation of new DAQ, seems that saturation occurs earlier now
        if params.stime22 > 0:
            fFullOverlap = 400.  # overlap has moved, after installation of new DAQ, seems that saturation occurs earlier now
        if params.stime232 > 0:
            fFullOverlap = params.gkFullOverlap6  # overlap has moved, after installation of new DAQ, seems that saturation occurs earlier now
        if params.stime233 > 0:
            fFullOverlap = params.gkFullOverlap5  # overlap has moved, after installation of new DAQ, seems that saturation occurs earlier now
        if params.stime234 > 0:
            fFullOverlap = params.gkFullOverlap7  # overlap has moved, after installation of new DAQ, seems that saturation occurs earlier now
        if params.stime26 > 0:
            fFullOverlap = params.gkFullOverlap5  # overlap was corrected again, valid for only one night
        if params.stime261 > 0:
            fFullOverlap = params.gkFullOverlap7  # overlap got worse, probably screw loose
        if params.stime27 > 0:
            fFullOverlap = 1500.  # overlap got worse, probably screw loose
        if params.stime285 > 0:
            fFullOverlap = 1000.  # overlap got worse, probably screw loose
        range_max_clouds = 23000.

        zenith_radians = np.deg2rad(zenith)

        if (range_max - 100.) * np.cos(zenith_radians) < range_max_clouds:
            print("Reducing maximum range to search for clouds from", range_max_clouds, "to",
                 (range_max - 100.) * np.cos(zenith_radians), "Zd=", zenith)
            range_max_clouds = (range_max - 100.) * np.cos(zenith_radians)
        return C_0

    def apply_time_dependent_corrections(self, datime, c0, coszd, case_index=None):
        stimes = [(datime - switch_time).total_seconds() for switch_time in switch_times]
        Alphafit_corr = 0.0

    def apply_zenith_correction(self, c0, coszd):
        gkC_0ZenithCorr = 0.038
        zd_corr = np.log(1 - gkC_0ZenithCorr * (1 - coszd))
        c0 += 2 * zd_corr
        return True

    def apply_azimuth_correction(self, c0, zenith, azimuth, zd_az_corr):
        zenith_threshold = zd_az_corr.correction_rules[0].get('zenith_threshold', None)
        azimuth_threshold = zd_az_corr.correction_rules[0].get('azimuth_threshold', None)
        zenith_threshold_min = zd_az_corr.correction_rules[0].get('zenith_threshold_min', None)
        zenith_threshold_max = zd_az_corr.correction_rules[0].get('zenith_threshold_max', None)
        azimuth_threshold_min = zd_az_corr.correction_rules[0].get('azimuth_threshold_min', None)
        azimuth_threshold_max = zd_az_corr.correction_rules[0].get('azimuth_threshold_max', None)

        if zenith_threshold is not None and azimuth_threshold is not None:
            if (zenith < zenith_threshold * u.deg and
                    azimuth_threshold[0] * u.deg < azimuth < azimuth_threshold[1] * u.deg):
                c0 -= 2 * zd_az_corr.correction_rules[0]['correction_factor']
                return True
        elif zenith_threshold_min is not None and zenith_threshold_max is not None:
            if (zenith_threshold_min * u.deg < zenith < zenith_threshold_max * u.deg and
                    azimuth_threshold_min * u.deg < azimuth < azimuth_threshold_max * u.deg):
                c0 -= 2 * zd_az_corr.correction_rules[0]['correction_factor']
                return True
        return False

    def get_bin_width(self, ifadc, bgsamples, ncollapse):
        #fBGSamples = 0
        if bgsamples <= params.gkBGSamplesV1to4:
            return 1.0
        if ifadc < 128:
            return 0.5
        if ifadc < 256:
            return 1.0
        if ifadc < 384:
            return 2.0
        if ifadc < 448:
            return 4.0
        return 4.0 * ncollapse

    def get_offset_r(self, ifadc, bgsamples, ncollapse):
        #fBGSamples = 0
        if bgsamples <= params.gkBGSamplesV1to4:
            return 0.0
        if ifadc < 128:
            return - bgsamples * params.gkIntegrationWindow * self.get_bin_width(ifadc, bgsamples-1, ncollapse)
        r = (128 - bgsamples) * params.gkIntegrationWindow * self.get_bin_width(ifadc, bgsamples-1, ncollapse)
        if ifadc < 256:
            return r
        r += 128 * params.gkIntegrationWindow
        if ifadc < 384:
            return r
        r += 128 * params.gkIntegrationWindow * 2
        if ifadc < 448:
            return r
        r += 64 * params.gkIntegrationWindow * 4
        return r

    def get_fadcr(self, i, bgsamples):
        #fBGSamples = 0
        if bgsamples <= params.gkBGSamplesV1to4:
            return i
        ifadc = i + bgsamples
        if ifadc < 448:
            return ifadc % 128
        return ifadc % 64

    def get_collapsed_signal(self, transmission6km, background1, phecounts, bg, bgvar):
        signalsamples = 0
        sig = np.zeros(signalsamples)
        if transmission6km < -998.:
            for i in range(466, signalsamples):
                background1 += np.power((phecounts[i] - phecounts[i - 1]) * 1E6 / np.power(params.gkIntegrationWindow * i, 2), 2)
            background1 /= 30.0
            bg = background1
            for i in range(signalsamples):
                sig[i] = pheounts[i] * 1E6 / np.power(params.gkIntegrationWindow * i, 2) + background1
        return sig

    def get_range(self, i, bgsamples, ncollapse, t0shift):
        ifadc = i + bgsamples
        return (self.get_offset_r(ifadc, bgsamples, ncollapse) + params.gkIntegrationWindow * (self.get_fadcr(i, bgsamples) + 0.5) * self.get_bin_width(ifadc, bgsamples, ncollapse)
            + t0shift * params.gkIntegrationWindow * self.get_bin_width(ifadc, bgsamples, ncollapse))

    def get_signal(self, i, bgsamples, phecounts, ncollapse, bg, hwcorr):
        ifadc = i + bgsamples
        return ((phecounts - self.get_bin_width(ifadc, bgsamples, ncollapse) * bg) * hwcorr
            / self.get_bin_width(ifadc, bgsamples, ncollapse))

def main():
    test_data = Path(os.getenv("MAGIC_TEST_DATA", "test_data")).absolute()
    test_calibrated_real_dir = test_data / "real/calibrated"
    input_file = (test_calibrated_real_dir/"20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root")
    report = ReportLaser()
    laser_params = report.magic_reports_reading(input_file)

    time = datetime.datetime(2020, 7, 26, 3, 10, 0)
    zenith = laser_params['Zenith']
    azimuth = laser_params['Azimuth']
    range_max = laser_params['RangeMax']
    phecounts = laser_params['PheCounts']
    bgsamples = laser_params['BGSamples']
    ncollapse = laser_params['NCollapse']
    transmission6km = laser_params['Transmission6km']
    background1 = laser_params['Background1']
    t0shift = laser_params['T0Shift']
    coszd = np.cos(np.deg2rad(zenith))
    ifadc = 200
    bg = 0.0
    bgvar = 0.0

    C_0 = report.get_c0(time, zenith, range_max)
    print("C_0 value:", C_0)

    case_index = 15
    report.apply_time_dependent_corrections(time, C_0, zenith, case_index)

    c0 = report.apply_zenith_correction(C_0, coszd)
    print("Zenith corrected c0:", c0)

    c0 = report.apply_azimuth_correction(C_0, zenith, azimuth, zd_az_corr)

    print("Zenith and Azimuth corrected c0:", c0)

    bin_width = report.get_bin_width(ifadc, bgsamples, ncollapse)
    print("Bin Width:", bin_width)

    offset_r = report.get_offset_r(ifadc, bgsamples, ncollapse)
    print("Offset R:", offset_r)

    fadc_r = report.get_fadcr(150, bgsamples)
    print("FADCR:", fadc_r)

    collapsed_signal = report.get_collapsed_signal(transmission6km, background1, phecounts, bg, bgvar)

    print("Collapsed Signal:", collapsed_signal)
    print("Background:", bg)
    print("Background Variance:", bgvar)

    range_value = report.get_range(0, bgsamples, ncollapse, t0shift)
    print("Range:", range_value)

if __name__ == "__main__":
    main()

