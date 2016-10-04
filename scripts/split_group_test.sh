#!/usr/bin/env bash

if [ -z ${dataset+x} ]; then
  export dataset="/file1/cifar10/split_group_test.t7"
fi
if [ -z ${model+x} ]; then
  export model="split_group_test"
fi
export learningRate=0.1
export epoch_step="{20,40,60}"
export max_epoch=80
export learningRateDecay=0
export learningRateDecayRatio=0.2
export nesterov=true
export randomcrop_type=reflection

# tee redirects stdout both to screen and to file
# have to create folder for script and model beforehand
for i in {1..5}; do
  export save=logs/${model}_${RANDOM}${RANDOM}
  mkdir -p $save
  export depth=$i
  echo "Training model with depth $depth"
  th train.lua | tee $save/log.txt
done
