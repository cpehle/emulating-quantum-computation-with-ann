# Quantum Harakiri

This repository contains code for reproducing the experiments reported in [fill in blank].

## Installing all Requirements

Install tensorflow-1.10 on macOS this can be accomplished by running
```
pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.0-py3-none-any.whl
```
All other requirements can be installed by running.
```
pip install -r requirements.txt
```
note that in order to use a GPU accelerated version of tensorflow / keras you need
to have the proper CUDA drivers and library installed.

To install NEST a script is provided in the ci/ folder, which will build a local copy
of the NEST master branch.