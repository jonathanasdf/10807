# Installation

The code depends on Torch http://torch.ch. Follow instructions [here](http://torch.ch/docs/getting-started.html).

## Training Example

```bash
th Train.lua "models/cifar.lua -channels 8 -nodedepth 1 -pool 3,6 CifarDeepTreeProcessor.lua -flip 0.5 -minCropPercent 0.8" /data/cifar10/trainval.txt out/output.t7 -val /data/cifar10/test.txt -valSize -1 -valEvery 1 -batchSize 128 -epochSize -1 -epochs 100 -learningRate 100 -LRDropEvery 20 -LRDropFactor 5
```

## Testing Example

```bash
th Forward.lua "out/output.t7 CifarDeepTreeProcessor.lua" /data/cifar10/test.txt -batchSize 128
```
