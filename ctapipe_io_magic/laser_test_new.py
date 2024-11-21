import datetime
import numpy as np
import astropy.units as u
from scipy.stats import t
from scipy.stats import iqr
from scipy.optimize import curve_fit
from ctapipe_io_magic import MAGICEventSource
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

class MReportLaser:
    def __init__(self):
        self.gkRangeMax2 = 31000.
        self.gkShotsNewLaser = 25000
        self.gkShotsOldLaser = 50000
        self.fRangeMax = 0.0
        self.fShots = 0
        self.fRangeMax_Clouds = 0.0
        self.fZenith = 50.0 * u.deg
        self.fAzimuth = 200.0 * u.deg
        self.fLog = None
        self.Alphafit_corr = 0.

        self.gkIntegrationWindow = 47.954
        self.gkHWSwitchV1to4 = 96
        self.gkSecsPerYear = 3600.*24*365
        self.gkC_0ZenithCorr = 0.038
        self.gkBins = 512
        self.gkMaxCloudLayers = 3
        self.gkUpdate1Vers = 201206050
        self.gkUpdate2Vers = 201302221 
        self.gkUpdate3Vers = 201304250 
        self.gkUpdate4Vers = 201401091 
        self.gkUpdate5Vers = 201705110
        self.gkC_0Elterman = 0.081
        self.coszd = np.cos(self.fZenith)

        """
        Code completion, need to decide which part of it should be kept
        """

        self.fCloudHM = [0.] * self.gkMaxCloudLayers
        self.fCloudHStd = [0.] * self.gkMaxCloudLayers
        self.fCloudLR = [0.] * self.gkMaxCloudLayers
        self.fCloudFWHM = [0.] * self.gkMaxCloudLayers
        self.fCloudBase = [0.] * self.gkMaxCloudLayers
        self.fCloudTop = [0.] * self.gkMaxCloudLayers
        self.fCloudTopT = [0.] * self.gkMaxCloudLayers
        self.fCloudTrans = [1.] * self.gkMaxCloudLayers
        self.fPheCounts = [0.] * self.gkBins
        self.fCalimaness = 0.
        self.fCloudLayerAlt = 0.
        self.fCloudLayerDens = 0.
        self.fTransmission3km = 0.
        self.fTransmission6km = 0.
        self.fTransmission9km = 0.
        self.fTransmission12km = 0.
        self.fErrorCode = 0.
        self.gkBGSamplesV1to4 = 16
        self.gkBGSamplesV5 = 31
        self.gkSignalSamplesV5 = 480
        self.gkHWSwitchV5 = 256
        self.fBGSamples = self.gkBGSamplesV5
        self.fSignalSamples = self.gkSignalSamplesV5
        self.fHWSwitch = self.gkHWSwitchV5
        self.gkC_0TempCorr3 = 0.0002
        self.temp_mean = 8.89
        self.gkC_0HumCorr = 0.00023

        self.ifadc = 1
        self.collapse = 1

        self.signalsamples = self.fSignalSamples
        self.fIsBGCorrection = 0
        self.fBackground1 = 0.0
        self.fBackground2 = 0.0
        self.fBackgroundErr1 = 0.0
        self.fBackgroundErr2 = 0.0
        self.fNCollapse = 4

    def magic_reports_reading(self, input_file, process_run=False):
        """Read laser parameters from MAGIC files"""
        logger.info(f"\nInput file: {input_file}")
        event_source = MAGICEventSource(input_file, process_run=process_run)
        laser = event_source.laser
        weather = event_source.weather
        return  laser, weather #{'laser': laser, 'weather': weather}

    def reset_clouds(self):
        for arr in (self.fCloudHM, self.fCloudHStd, self.fCloudLR, self.fCloudFWHM,
                    self.fCloudBase, self.fCloudTop, self.fCloudTopT, self.fCloudTrans):
            arr[:] = [0.] * self.gkMaxCloudLayers

    def interprete_body(self, str, ver):
        interpreters = {
            self.gkUpdate1Vers: self.interprete_clouds,
            self.gkUpdate4Vers: self.interprete_cloud_base,
            self.gkUpdate3Vers: self.interprete_error_code,
            self.gkUpdate1Vers: self.interprete_pointing,
        }
        for i in range(self.gkBins):
            n = sscanf(str, "%08x" if ver >= self.gkUpdate1Vers else "%06x")
            if n != 1:
                self.fLog << _warn_ << GetDescriptor() << f": could not read photon counts for bin {i}, instead got: {str}" << endl
                return kCONTINUE
            str = str[6 if ver < self.gkUpdate1Vers else 8:]

        for v, interpreter in interpreters.items():
            if ver >= v:
                if not interpreter(str):
                    return kCONTINUE

        if str != "OVER":
            self.fLog << _warn_ << f"WARNING - 'OVER' tag not found (instead: {str})" << endl
        return kTRUE

    def interprete_clouds(self, str):
        self.fCalimaness, self.fCloudLayerAlt, self.fCloudLayerDens, str = map(float, str.split(' ', 3))
        return kTRUE

    def interprete_cloud_base(self, str):
        self.fCloudLayerAlt, str = map(float, str.split(' ', 1))
        return kTRUE

    def interprete_error_code(self, str):
        self.fErrorCode, str = map(int, str.split(' ', 1))
        return kTRUE

    def interprete_pointing(self, str):
        self.fZenith, self.fAzimuth, str = map(float, str.split(' ', 2))
        return kTRUE

    def print(self, o):
        self.fLog << GetDescriptor() << ":" << endl
        for i, phe in enumerate(self.fPheCounts):
            self.fLog << f"i={i}    binw: {GetBinWidth(i):.0f}    range: {(GetOffsetR(i) + gkIntegrationWindow * (GetFADCR(i - self.fBGSamples) + 0.5) * GetBinWidth(i)) * 0.001:.2f} km    counts: {phe / GetBinWidth(i)}" << endl

        self.fLog << endl
        self.fLog << f"Calimaness: {self.fCalimaness}, Altiude of first cloud layer: {self.fCloudLayerAlt}, Optical thickness cloud layer: {self.fCloudLayerDens}" << endl
        self.fLog << f"Online analysis: Transmission3km: {self.fTransmission3km}, Transmission6km: {self.fTransmission6km}, Transmission9km: {self.fTransmission9km}, Transmission12km: {self.fTransmission12km}" << endl
        self.fLog << endl

    def eval_gdas(self, x, par):
        hlo, costheta, logPTMAGIC, h = par[1:]
        return par[0] - logPTMAGIC + self.fGDAS.EvalF(hlo, h, costheta, fBeta_mol_0)

    def read_switch_times(self, filename):
        overlaps = {}
        C_0s = {}
        with open(filename, 'r') as file:
            for line in file:
                if line[0] != "#":
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        dt_str = ' '.join(parts[:2])
                        overlap = float(parts[2].rstrip('.'))
                        C_0 = float(parts[3])
                        self.switch_time = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                        overlaps[self.switch_time] = overlap
                        C_0s[self.switch_time] = C_0
        return overlaps, C_0s

    def read_zd_az_conditions(self, filename):
        with open(filename, 'r') as file:
            azimuth_conditions = [list(map(float, line.strip().split())) for line in file if not line.startswith('#')]
        return azimuth_conditions
    
    """
    C_0 parameter calculation based on time
    """
    def GetC0(self, time, overlaps, C_0s):
        if time:
            for switchtime, overlap in overlaps.items():
                stime = int((time - switchtime).total_seconds())
                if stime > 0:
                    FullOverlap = overlap
                    C_0 = C_0s[switchtime]

        self.fRangeMax = 40000. if self.fZenith > 70. * u.deg else self.gkRangeMax2
        self.fShots = self.gkShotsNewLaser
        self.fRangeMax_Clouds = 23000.
        
        self.fRangeMax_Clouds = np.min([(self.fRangeMax - 100.) * np.cos(self.fZenith), self.fRangeMax_Clouds])

        return C_0, FullOverlap

    def ApplyZenithCorrection(self, c0):
        print("C_0_prev:", c0)
        zd_corr = np.log(1 - self.gkC_0ZenithCorr * (1 - self.coszd))
        c0 += 2 * zd_corr
        return True

    def ApplyAzimuthCorrection(self, c0, zd_az_conditions):
        for condition in zd_az_conditions: 
            zenith_min, zenith_max, azimuth_min, azimuth_max, correction = condition
            if (zenith_min <= self.fZenith.value <= zenith_max and
                azimuth_min <= self.fAzimuth.value <= azimuth_max):
                c0 -= 2 * correction
                return True
        return False

    def ApplyTimeCorrection(self, c0, time, c1, c2, zd_az_conditions):
        if time:
            for switchtime, corr1 in c1.items():
                stime = int((time - switchtime).total_seconds())
                if stime < 0:
                    correction1 = corr1
                    correction2 = c2[switchtime]
                    if (time - switchtime28).total_seconds() < 0 < (time - switchtime275).total_seconds():
                        self.ApplyAzimuthCorrection(c0, zd_az_conditions)
                    c0 -= 2 * (correction1 + correction2 * prev_stime / self.gkSecsPerYear) + 2.0 / self.coszd * self.Alphafit_corr
                    return c0
                prev_stime = stime
        return c0

    def apply_temperature_correction(self, c0, temperature):
        """valid since 17.11.2016"""
        c0 -= self.gkC_0TempCorr3 * (temperature - self.temp_mean)

        return True, c0

    def apply_humidity_correction(self, c0, humidity):
        if humidity < 0:
            return False

        c0 -= self.gkC_0HumCorr * (humidity - 30.0)

        return True, c0

    def test_c0_drift(self, c1, c2, zd_az_conditions, filename):
        time0 = datetime.datetime(2019, 1, 1, 0, 0, 0)
        times = []
        c0_values = []
        with open(filename, 'w') as file:
            for i in range(int(5.5 * 365)):
                c0 = 0.
                time1 = time0 + datetime.timedelta(days=i)
                c0 = self.ApplyTimeCorrection(c0, time1, c1, c2, zd_az_conditions)
                times.append(time1)
                c0_values.append(c0)
                file.write(f"{time1.year}-{time1.month}-{time1.day} {time1.hour}:{time1.minute} {c0}\n")
                print(f"{time1.year}-{time1.month}-{time1.day} {time1.hour}:{time1.minute} {c0:.5f}")
        plt.rcParams['font.size'] = 20
        fig, ax = plt.subplots()
        ax.plot(times, c0_values, marker='o', linestyle='-', label='C0 Drift')
        
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.set_xlabel('Time')
        ax.set_ylabel('C_0 Correction')
        #ax.legend()
        plt.xticks() #(rotation=45)
        plt.tight_layout()
        plt.savefig("c0_python.png")
        plt.show()

# Sample usage:
def main():
    input_file = '/home/zywuckan/ctapipe_io_magic/ctapipe_io_magic/tests/test_data/real/calibrated/20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root'
    report = MReportLaser()
    laser_list, weather_list = report.magic_reports_reading(input_file)  # Unpack the tuple
    first_laser = laser_list[0]
    first_weather = weather_list[0]
    time = datetime.datetime(2020, 7, 26, 3, 10, 0)
    zenith = first_laser.Zenith #['Zenith']
    azimuth = first_laser['Azimuth']
    temperature = first_weather['Temperature'] #* u.deg_C
    print(temperature)
    humidity = first_weather['Humidity']
    print(humidity)
    i = 0
    overlaps, C_0s = report.read_switch_times('switchtimes1.txt')
    c0, _ = report.GetC0(time, overlaps, C_0s)
    c0, overlap = report.GetC0(time, overlaps, C_0s)
    print("C0:", c0)
    print("overlap:", overlap)

    zenith_corr = report.ApplyZenithCorrection(c0)
    print(zenith_corr)

    zd_az_conditions = report.read_zd_az_conditions('azimuth_conditions.txt')
    azimuth_corr = report.ApplyAzimuthCorrection(c0, zd_az_conditions)
    print("azimuth correction:", azimuth_corr)

    c1, c2 = report.read_switch_times('switch_times.txt')
    time_corr2 = report.ApplyTimeCorrection(c0, time, c1, c2, zd_az_conditions)
    print("time correction:", time_corr2)

    temp = report.apply_temperature_correction(c0, temperature)
    print(temp)

    humid = report.apply_humidity_correction(c0, humidity)
    print(humid)

    report.test_c0_drift(c1, c2, zd_az_conditions, 'c0_drift_data.txt')

if __name__ == "__main__":
    main()
