# Preconditioners for the Stochastic Training of Neural fields (CVPR-2025)
[Shin-Fang Chng](https://sfchng.github.io)\*,
[Hemanth Saratchandran]()\*,
[Simon Lucey]() <br>
Australian Institute for Machine Learning (AIML), University of Adelaide, \* denotes equal contribution


This is the official implementation of the paper "Preconditioners for the Stochastic Training of Neural fields".

## Setup ##

## Installation ##

## Datasets
### Div2k data ###
We use the ``div2k`` dataset for our 2d image experiment. Please download the dataset [here](https://universityofadelaide.box.com/s/13twlttg9aagf4srye11c6oh41t04dv5), and place it under
the directory ``data``.

### Stanford data ##
We use the ``stanford`` dataset for our 3d binary occupancy experiment. Please download the dataset [here](https://universityofadelaide.box.com/s/k435ov4uoj8pybzdunuc3m92gap14zjp), and place it under the directory ``data/bocc``.

### Run commands ###
```
# Image experiment
./scripts/neural_image.sh
```
```
# Binary occupancy experiment
```

### Key results ###
ESGD (a Curvature-aware preconditioned gradient descent algorithm) improves convergence for Gaussian, sine and wavelet activations, while Adam performs
better for ReLU network with positional encoding (ReLU(PE)). We provide training convergence for a 2D image reconstruction task as an example below
<p align="center" width="100%">
<img src="misc/gaussian_convergence.png" width="40%"> <img src="misc/sine_convergence.png" width="40%"> 
<img src="misc/wavelet_convergence.png" width="40%"> <img src="misc/relu_convergence.png" width="40%"> 
</p>
