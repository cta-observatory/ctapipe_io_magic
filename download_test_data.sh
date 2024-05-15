#!/bin/bash

set -e

echo "https://www.magic.iac.es/mcp-testdata/test_data/real/calibrated/20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root" >  test_data_real.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/real/calibrated/20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root" >> test_data_real.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/real/calibrated/20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root" >> test_data_real.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/real/calibrated/20210314_M2_05095172.002_Y_CrabNebula-W0.40+035.root" >> test_data_real.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/real/calibrated/20230324_M1_05106879.001_Y_1ES0806+524-W0.40+000.root" >>  test_data_real.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/real/calibrated/20230324_M1_05106879.002_Y_1ES0806+524-W0.40+000.root" >>  test_data_real.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/real/calibrated/20230324_M2_05106879.001_Y_1ES0806+524-W0.40+000.root" >>  test_data_real.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/real/calibrated/20230324_M2_05106879.002_Y_1ES0806+524-W0.40+000.root" >>  test_data_real.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/real/calibrated/20210314_M1_05095172.001_Y_CrabNebula-W0.40+035_only_events.root" >  test_data_real_missing_trees.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/real/calibrated/20210314_M1_05095172.001_Y_CrabNebula-W0.40+035_only_drive.root" >>  test_data_real_missing_trees.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/real/calibrated/20210314_M1_05095172.001_Y_CrabNebula-W0.40+035_only_runh.root" >>  test_data_real_missing_trees.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/real/calibrated/20210314_M1_05095172.001_Y_CrabNebula-W0.40+035_only_trigger.root" >>  test_data_real_missing_trees.txt

echo "https://www.magic.iac.es/mcp-testdata/test_data/real/calibrated/20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root" > test_data_real_missing_prescaler_trigger.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/real/calibrated/20210314_M1_05095172.002_Y_CrabNebula-W0.40+035_no_prescaler_trigger.root" >> test_data_real_missing_prescaler_trigger.txt

echo "https://www.magic.iac.es/mcp-testdata/test_data/real/calibrated/20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root" > test_data_real_missing_arrays.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/real/calibrated/20210314_M1_05095172.002_Y_CrabNebula-W0.40+035_no_arrays.root" >> test_data_real_missing_arrays.txt

echo "https://www.magic.iac.es/mcp-testdata/test_data/simulated/calibrated/GA_M1_za35to50_8_824318_Y_w0.root" >  test_data_simulated.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/simulated/calibrated/GA_M1_za35to50_8_824319_Y_w0.root" >> test_data_simulated.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/simulated/calibrated/GA_M2_za35to50_8_824318_Y_w0.root" >> test_data_simulated.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/simulated/calibrated/GA_M2_za35to50_8_824319_Y_w0.root" >> test_data_simulated.txt

echo "https://www.magic.iac.es/mcp-testdata/test_data/real/superstar/20210314_05095172_S_CrabNebula-W0.40+035.root" > test_data_superstar_real.txt

echo "https://www.magic.iac.es/mcp-testdata/test_data/simulated/superstar/GA_za35to50_8_824318_S_w0.root" > test_data_superstar_simulated.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/simulated/superstar/GA_za35to50_8_824319_S_w0.root" >> test_data_superstar_simulated.txt

echo "https://www.magic.iac.es/mcp-testdata/test_data/real/melibea/20210314_05095172_Q_CrabNebula-W0.40+035.root" > test_data_melibea_real.txt

echo "https://www.magic.iac.es/mcp-testdata/test_data/simulated/melibea/GA_za35to50_8_824318_Q_w0.root" > test_data_melibea_simulated.txt
echo "https://www.magic.iac.es/mcp-testdata/test_data/simulated/melibea/GA_za35to50_8_824319_Q_w0.root" >> test_data_melibea_simulated.txt

if [ -z "$TEST_DATA_USER" ]; then
    read -p "Username: " TEST_DATA_USER
    echo
fi

if [ -z "$TEST_DATA_PASSWORD" ]; then
    read -sr -p "Password: " TEST_DATA_PASSWORD
    echo
fi

declare -A TEST_FILES_DOWNLOAD

TEST_FILES_DOWNLOAD[test_data_real]="test_data/real/calibrated"
TEST_FILES_DOWNLOAD[test_data_real_missing_trees]="test_data/real/calibrated/missing_trees"
TEST_FILES_DOWNLOAD[test_data_real_missing_prescaler_trigger]="test_data/real/calibrated/missing_prescaler_trigger"
TEST_FILES_DOWNLOAD[test_data_real_missing_arrays]="test_data/real/calibrated/missing_arrays"
TEST_FILES_DOWNLOAD[test_data_simulated]="test_data/simulated/calibrated"
TEST_FILES_DOWNLOAD[test_data_superstar_real]="test_data/real/superstar"
TEST_FILES_DOWNLOAD[test_data_melibea_real]="test_data/real/melibea"
TEST_FILES_DOWNLOAD[test_data_superstar_simulated]="test_data/simulated/superstar"
TEST_FILES_DOWNLOAD[test_data_melibea_simulated]="test_data/simulated/melibea"

for key in "${!TEST_FILES_DOWNLOAD[@]}"
do
    if ! wget \
        -i "${key}.txt" \
        --user="$TEST_DATA_USER" \
        --password="$TEST_DATA_PASSWORD" \
        --no-check-certificate \
        --no-verbose \
        --timestamping \
        --directory-prefix="${TEST_FILES_DOWNLOAD[${key}]}"; then
    echo "Problem in downloading the test data set from ${key}.txt."
fi
done

rm -f test_data_real.txt test_data_simulated.txt test_data_real_missing_trees.txt test_data_real_missing_prescaler_trigger.txt test_data_real_missing_arrays.txt test_data_superstar_real.txt test_data_superstar_simulated.txt test_data_melibea_real.txt test_data_melibea_simulated.txt
