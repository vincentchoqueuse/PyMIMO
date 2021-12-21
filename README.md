# Deep MIMO Detection

## Summary

The PyMIMO is a python library for testing MIMO Communications. The library is composed of several modules

* `processors`: classical processors for digital communication (modulation, ...)
* `channels`: Static and Gaussian Frequency Flat Channels 
* `detectors`: classical MIMO Detectors (ML, ZF, MMSE) and Deep Learning based detector (FullyCon)

## Requirements

This library is based upon the following libraries: Numpy, Scipy, Pandas, PyTorch, Plotly (see `requirements.txt`). To install the following libraries, please run the command

```
pip install -r requirements.txt 
```

## List of simulations

The `tests` folder contains several scripts for testing MIMO detector performances

* SER performance of ML, ZF and MMSE detectors over Gaussian Channels

```
cd tests
python test_detectors.py
```

* SER performance of ZF and DL detectors over Statitic Channels

```
cd tests
python test_DL_detectors.py
```

## Documentation

The documentation is available in the `/doc` folder