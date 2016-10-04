# Installation

The code depends on Torch http://torch.ch. Follow instructions [here](http://torch.ch/docs/getting-started.html) and run:

```
luarocks install torchnet
```

We recommend installing CUDNN v5 for speed. Alternatively you can run on CPU.

For visualizing training curves we used ipython notebook with pandas and bokeh and suggest using anaconda.

# Usage

## Dataset support

The code supports loading simple datasets in torch format. We provide the following:

* CIFAR-10 whitened (using pylearn2) [preprocessed dataset](https://yadi.sk/d/em4b0FMgrnqxy)

To whiten CIFAR-10 and CIFAR-100 we used the following scripts https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/datasets/make_cifar10_gcn_whitened.py and then converted to torch using https://gist.github.com/szagoruyko/ad2977e4b8dceb64c68ea07f6abf397b and npy to torch converter https://github.com/htwaijry/npy4th.

## Training

```bash
model=<model> dataset=/file1/cifar10/data.t7 scripts/train_cifar.sh
```
For reference we provide logs for this experiment and [ipython notebook](notebooks/visualize.ipynb) to visualize the results.

Multi-GPU is supported with `nGPU=n` parameter.
