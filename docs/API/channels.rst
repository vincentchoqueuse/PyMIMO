MIMO Channels
=============

In a Frequency-Flat MIMO Channel. The processor output is given by 

.. math ::

    \mathbf{y} = \mathbf{H}\mathbf{x}+\mathbf{b}

* :math:`\mathbf{H}\in \mathcal{C}^{N_r\times N_t}`: channel matrix,
* :math:`\mathbf{x}\in \mathcal{M}^{N_t}`: transmitted symbols
* :math:`\mathbf{y}\in \mathcal{C}^{N_r}`: received samples
* :math:`\mathbf{b}\sim \mathcal{N}(0,\sigma^2 \mathbf{I})`: additive white Gaussian noise with zero-mean and variance :math:`sigma^2`



List of Channels
----------------

.. automodule:: PyMIMO.channels
    :members:
    :imported-members: