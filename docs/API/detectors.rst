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
* :math:`\mathbf{b}\in \mathcal{C}^{N_r}`: additive noise

A MIMO detector detects the transmitted symbols from the received data. Mathematically, the 
detected symbols, :math:`\widehat{\mathbf{x}}`, can be expressed as

.. math :: 
    
    \widehat{\mathbf{x}}=\mathbf{f}(\mathbf{y})

where :math:`\mathbf{f}(.)` is a non-linear function that depends on the chosen detector.

Maximum Likelihood Detectors 
----------------------------

For white Gaussian noise :math:`\mathbf{b}\sim \mathcal{N}(0,\sigma^2\mathbf{I})`, 
the Maximum Likelihood Detectors is given by

.. math :: 

        \widehat{\mathbf{x}} = \arg \min_{\mathbf{x}\in \mathcal{M}^{N_t}}\|\mathbf{y}-\mathbf{H}\mathbf{x}\|^2_2
    
where :math:`\mathcal{M}` is the constellation of the transmitted data. 

.. note ::

   The ML detector has a prohibitive computional complexity for large number :math:`N_t` since the set :math:`\mathbf{x}\in \mathcal{M}^{N_t}` is composed of :math:`M^{N_t}` elements with :math:`M=|\mathcal{M}|`.

Linear Detectors 
----------------

To reduce the computational complexity, linear detectors decompose the detection problem into two subproblems:

* Linear estimation of the transmitted symbols:

.. math :: 

    \widehat{\mathbf{x}}_0 = \mathbf{W}\mathbf{y}

* Element-wise nonlinear projection of the estimated symbols into the constellation set as

.. math :: 

    \widehat{x}[k] = \mathcal{P}(\mathbf{x}_0[k]) =\arg \min_{x\in \mathcal{M}}|x-\widehat{x}_0[k]|^2_2


Zero Forcing Detector
+++++++++++++++++++++

In the estimation step, the ZF detector solves the following problem

.. math ::

    \widehat{\mathbf{x}}_0 = \arg \min_{\mathbf{x}} \|\mathbf{y}-\mathbf{H}\mathbf{x}\|^2_2

The solution is given by the matrix :

.. math :: 

    \mathbf{W} = \mathbf{H}^{\dagger}

where :math:`\mathbf{H}^{\dagger}=\left(\mathbf{H}^H\mathbf{H}\right)^{-1}\mathbf{H}^H` corresponds to the matrix pseudoinverse of the channel matrix. 

MMSE Detector 
+++++++++++++

In the estimation step, the MMSE detector solves the following problem

.. math ::

    \widehat{\mathbf{x}}_0 = \arg \min_{\mathbf{x}} \|\mathbf{y}-\mathbf{H}\mathbf{x}\|^2_2+\sigma^2\|\mathbf{x}\|^2_2

The solution is given by the matrix :

.. math ::

    \mathbf{W} = \left(\mathbf{H}^{H}\mathbf{H}+\sigma^2\mathbf{I} \right)^{-1}\mathbf{H}^H \mathbf{y}
        


List of Detectors 
-----------------

.. automodule:: PyMIMO.detectors
    :members:
    :imported-members:
    :show-inheritance:
    :inherited-members: