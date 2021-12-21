Getting Started
===============

Install the dependencies
------------------------

First, be sure that the following libraries are properly installed 

* Pandas 
* Plotly
* Pytorch 

First Tests 
-----------

The easiest to start with this library is to run some test python scripts. The folder `test` 
contains several python script for testing the performance of classical detector.

.. code ::

    cd tests
    python test_detectors.py

For example, the script file `test_detectors.py` performs a monte carlo simulation to approximate the SER detection performance
of the the Maximum Likelihood (ML), the Zero-Forcing, and the MMSE detectors. The simulation parameters (modulation, channel, ...) are provided in the 
`simulation1.ini` file. After simulation, a plotly graph is automatically displayed on your default web browser. The simulation results are also saved in the csv folder.

