import numpy as np
from scipy.stats import norm
from ..core import Processor

class Static_Channel(Processor):

    """Implements a Static Frequency-Flat MIMO Channel :math:`\mathbf{H}`.

    Parameters
    ----------
    H : numpy array
        Channel matrix
    sigma2 : float
        noise variance

    """

    def __init__(self,H,sigma2=0):
        super().__init__()
        self._H = H
        self._sigma2 = sigma2
        self._N_r = H.shape[0]

    def set_SNR(self,SNR):
        """ Set the SNR (dB) """ 
        H_H = np.transpose(np.conjugate(self._H))
        sigma2_s = np.trace(np.matmul(self._H,H_H))/self._N_r
        self._sigma2 = sigma2_s/10**(SNR/10) 

    def forward(self,X):
        N_t, N = X.shape
        B = np.sqrt(self._sigma2/2)*(norm.rvs(size=(self._N_r,N))+1j*norm.rvs(size=(self._N_r,N)))
        Y = np.matmul(self._H,X) + B
        return Y

class Gaussian_Channel(Processor):

    """Implements a Gaussian Frequency-Flat MIMO Channel with elements

    .. math ::

        h_{u,v} \sim \mathcal{N}(0,\sigma_h^2)
    
    Parameters
    ----------
    N_r : float
        Number of received samples
    
    sigma2 : float
        noise variance

    """

    def __init__(self,N_r,sigma2_h=1,sigma2_s=1,sigma2=0):
        super().__init__()
        self._N_r = N_r
        self._sigma2_h = sigma2_h
        self._sigma2 = sigma2
        self._sigma2_s = sigma2_s
        self._snr_dB = None
        self._H = None

    def get_H(self):
        return self._H

    def get_sigma2(self):
        return self._sigma2

    def set_SNR(self,SNR):
        self._snr_dB = SNR

    def forward(self,X):
        N_t, N = X.shape

        self._H = np.sqrt(self._sigma2_h/2)*(norm.rvs(size=(self._N_r,N_t))+1j*norm.rvs(size=(self._N_r,N_t)))
        
        if self._snr_dB is not None:
            snr = 10**(self._snr_dB/10) 
            H_H = np.transpose(np.conjugate(self._H))
            self._sigma2 = self._sigma2_s*np.trace(np.matmul(self._H,H_H))/(self._N_r*snr)
        
        B = np.sqrt(self._sigma2/2)*(norm.rvs(size=(self._N_r,N))+1j*norm.rvs(size=(self._N_r,N)))
        Y = np.matmul(self._H,X) + B
        return Y