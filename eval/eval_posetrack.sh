#!/bin/bash
# Script to extract COCO JSON file for each trained model
#clear && clear

rm -rf posetrack_results
mkdir posetrack_results
mkdir posetrack_results/op_output

OPENPOSE_FOLDER=$(pwd)/../
POSETRACK_FOLDER=$(pwd)/posetrack/images/val
POSETRACK_JSON_FOLDER=$(pwd)/posetrack/annotations/val_json
#POSETRACK_FOLDER=$(pwd)/posetrack/images/val
#POSETRACK_JSON_FOLDER=$(pwd)/posetrack/annotations/test_json/

MODE=$1
MODEL="BODY_25B"
if [ -n "$2" ]; then
  MODEL=$2
fi

# Not coded for Multi GPU Yet
N=1
(
for folder in $POSETRACK_JSON_FOLDER/* ; do 

  # Setup name
  filename="${folder##*/}"
  filename="${filename%.*}"
  folder=$POSETRACK_FOLDER/$filename

  if [[ -d "$folder" && ! -L "$folder" ]]; then
    ((i=i%N)); ((i++==0)) && wait
    process=$((i%N));

    if [ "$MODE" = "tracking" ]; then 

      # Operation
      cd $OPENPOSE_FOLDER;
      ./build/examples/openpose/openpose.bin \
          --model_pose $MODEL --num_gpu 1 --num_gpu_start $process \
          --tracking 1 \
          --image_dir $folder \
          --write_json eval/posetrack_results/op_output/$filename \
          --render_pose 0 --display 0 > output.txt &
          #--render_pose 1 > output.txt &
          
    else

      # Operation
      cd $OPENPOSE_FOLDER;
      ./build/examples/openpose/openpose.bin \
          --model_pose $MODEL --num_gpu 1 --num_gpu_start $process \
          --image_dir $folder \
          --write_json eval/posetrack_results/op_output/$filename \
          --render_pose 0 --display 0 > output.txt &

    fi

    # sleep 1 &
    # echo "$folder $var is a directory" & 

  fi; 
done
)

sleep 10

if [ "$MODE" = "tracking" ]; then 
  python eval_posetrack.py $POSETRACK_JSON_FOLDER 1 $MODEL
else
  python eval_posetrack.py $POSETRACK_JSON_FOLDER 0 $MODEL
fi

# I need a script that makes it easy to automatically copy into OP to test given the name and model directory
# Video training is messing up completely..why? (appears to mess up with Posetrack Video datasets) - WHY!!?
# Test - Dont give Posetrack video, just give regular image?