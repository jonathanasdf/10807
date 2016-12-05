# Installation

The code depends on Torch http://torch.ch. Follow instructions [here](http://torch.ch/docs/getting-started.html).

## Example

### Standard convnet
```bash
th Train.lua "models/simple.lua CifarProcessor.lua -flip 0.5 -minCropPercent 0.8" /data/cifar10/trainval.txt out/output.t7 -val /data/cifar10/test.txt -valSize -1 -valEvery 1 -batchSize 128 -epochSize -1 -epochs 100 -learningRate 0.01 -LRDropEvery 30 -LRDropFactor 5
th Forward.lua "out/output.t7 CifarProcessor.lua" /data/cifar10/test.txt -batchSize 128
```

### Weighted version that doesn't work
```bash
th Train.lua "models/DeepTreeWeighted.lua -channels 64 -preconv 3 -depth 3 -nodedepth 1 -pool 3,5 CifarDeepTreeWeightedProcessor.lua -flip 0.5 -minCropPercent 0.8" /data/cifar10/trainval.txt out/output.t7 -val /data/cifar10/test.txt -valSize -1 -valEvery 1 -batchSize 128 -epochSize -1 -epochs 100 -learningRate 0.01 -LRDropEvery 30 -LRDropFactor 5
th Forward.lua "out/output.t7 CifarDeepTreeWeightedProcessor.lua" /data/cifar10/test.txt -batchSize 128
```

### With pretraining
```bash
th TrainDeepTreeWithPretraining.lua "models/DeepTreeRootOnly.lua -channels 64 -preconv 3 -depth 3 -nodedepth 1 -pool 3,5 CifarDeepTreeProcessor.lua -flip 0.5 -minCropPercent 0.8 [-greedy]" /data/cifar10/trainval.txt out/output.t7 -val /data/cifar10/test.txt -valSize -1 -valEvery 1 -batchSize 160 -epochSize -1 -pretrainEpochs 20 -epochs 100 -learningRate 0.01 -LRDropEvery 30 -LRDropFactor 5
th Forward.lua "out/output.t7 CifarDeepTreeProcessor.lua [-forwardWeighted]" /data/cifar10/test.txt -batchSize 128
```

### Optimizing full tree from random initialization
```bash
th Train.lua "models/DeepTree.lua -channels 64 -preconv 3 -depth 3 -nodedepth 1 -pool 3,5 CifarDeepTreeProcessor.lua -flip 0.5 -minCropPercent 0.8 [-greedy]" /data/cifar10/trainval.txt out/output.t7 -val /data/cifar10/test.txt -valSize -1 -valEvery 1 -batchSize 160 -epochSize -1 -pretrainEpochs 20 -epochs 100 -learningRate 0.01 -LRDropEvery 30 -LRDropFactor 5
th Forward.lua "out/output.t7 CifarDeepTreeProcessor.lua [-forwardWeighted]" /data/cifar10/test.txt -batchSize 128
```
