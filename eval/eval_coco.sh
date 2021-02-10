#!/bin/bash
# Script to extract COCO JSON file for each trained model
clear && clear

echo "User configurable options"
PERFORM_1_SCALE=YES
PERFORM_4_SCALES=NO
OPENPOSE_FOLDER=$(pwd)/../
TEST_2017_DEV=$(pwd)/val2017
RESULTS_FOLDER=$(pwd)/testing_results
TESTSET=test-val2017
ALGORITHM=OpenPose
MODEL=BODY_25B
echo " "

echo "Fix parameters"
ALGORITHM1=OpenPose1scale
ALGORITHM4=OpenPose4scales
FINAL_PATH1=${RESULTS_FOLDER}/person_keypoints_${TESTSET}_${ALGORITHM1}_results
FINAL_PATH4=${RESULTS_FOLDER}/person_keypoints_${TESTSET}_${ALGORITHM4}_results
FINAL_JSON_PATH1=${FINAL_PATH1}.json
FINAL_JSON_PATH4=${FINAL_PATH4}.json
FINAL_ZIP_PATH1=${FINAL_PATH1}.zip
FINAL_ZIP_PATH4=${FINAL_PATH4}.zip
TEMPORARY_JSON_FILE1=~/Desktop/temporaryJsonTest2017Dev_1.json
TEMPORARY_JSON_FILE4=~/Desktop/temporaryJsonTest2017Dev_4.json
echo " "





# # Change to OpenPose directory
echo "Running OpenPose on test2017-dev folder..."
cd $OPENPOSE_FOLDER
# Processing
if [ ${PERFORM_1_SCALE} = "YES" ]; then
    ./build/examples/openpose/openpose.bin \
        --model_pose $MODEL \
        --image_dir $TEST_2017_DEV \
        --write_coco_json $TEMPORARY_JSON_FILE1 \
        --render_pose 0 --display 0
    # Move JSON after finished (so no Dropbox updating all the time)
    mv $TEMPORARY_JSON_FILE1 $FINAL_JSON_PATH1
fi
# 4 scales max accuracy
if [ ${PERFORM_4_SCALES} = "YES" ]; then
    ./build/examples/openpose/openpose.bin \
        --image_dir $TEST_2017_DEV \
        --write_coco_json $TEMPORARY_JSON_FILE4 \
        --render_pose 0 --display 0 \
        --num_gpu 1 --scale_number 4 --scale_gap 0.25 --net_resolution "1312x736"
    # Move JSON after finished (so no Dropbox updating all the time)
    mv $TEMPORARY_JSON_FILE4 $FINAL_JSON_PATH4
fi
echo " "

# Zip file
echo "Zipping JSON file..."
cd $RESULTS_FOLDER
if [ ${PERFORM_1_SCALE} = "YES" ]; then
    zip $FINAL_ZIP_PATH1 $(basename ${FINAL_JSON_PATH1})
fi
if [ ${PERFORM_4_SCALES} = "YES" ]; then
    zip $FINAL_ZIP_PATH4 $(basename ${FINAL_JSON_PATH4})
fi
echo " "

# Run Test
cd ..
python eval_coco.py

echo "Finished! Exiting script..."
echo " "
