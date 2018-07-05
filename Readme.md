# Quantum Harakiri

This repository contains code for reproducing the experiments reported in [fill in blank].

## Installing all Requirements

Using pip all nescessary requirements except for NEST can be installed using the following command
```
pip install -r requirements.txt
```
note that in order to use a GPU accelerated version of tensorflow / keras you need
to have the proper CUDA drivers and library installed.

To install NEST a script is provided in the ci/ folder, which will build a local copy
of the NEST master branch.