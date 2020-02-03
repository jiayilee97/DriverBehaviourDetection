#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180619_kfold/lmdb_v3
DATA=/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180619_kfold/10_cross_validation_list_v2
TOOLS=build/tools

TRAIN_DATA_ROOT=/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180619_kfold/imgs/train/
VAL_DATA_ROOT=/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180619_kfold/imgs/train/
#TEST_DATA_ROOT=/media/ctg-sugiri-ia2/ac13d8c4-7f4c-46c6-b0ed-7fbc475ab744/ctg-sugiri-ia2/20180614_rcnn_mask/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train_kfold0.txt \
    $EXAMPLE/train_lmdb0
    


echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/val_kfold0.txt \
    $EXAMPLE/val_lmdb0




#echo "Creating test lmdb..."

#GLOG_logtostderr=1 $TOOLS/convert_imageset \
#    --resize_height=$RESIZE_HEIGHT \
#    --resize_width=$RESIZE_WIDTH \
#    --shuffle \
#    $TEST_DATA_ROOT \
#    $DATA/test.txt \
#    $EXAMPLE/test_lmdb

echo "Done."
