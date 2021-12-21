MIMO Detectors
==============


Problem Statement
-----------------

In a MIMO frequency flat channel, the output samples can be expressed as 

.. math ::

    \mathbf{y} = \mathbf{H}\mathbf{x}+\mathbf{b}

* :math:`\mathbf{H}\in \mathcal{C}^{N_r\times N_t}`: channel matrix,
* :math:`\mathbf{x}\in \mathcal{M}^{N_t}`: transmitted symbols
* :math:`\mathbf{y}\in \mathcal{C}^{N_r}`: received samples
* :math:`\mathbf{b}`: additive noise

A MIMO detector detects the transmitted symbols :math:`\widehat{\mathbf{x}}=\mathbf{f}(\mathbf{y})` from the received data :math:`\mathbf{y}`. 


List of Detectors 
-----------------

.. automodule:: PyMIMO.detectors
    :members:
    :imported-members: