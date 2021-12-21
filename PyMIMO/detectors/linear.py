import numpy as np
import numpy.linalg as LA
from ..core import Processor

class Linear_Detector(Processor):

    def __init__(self,H,alphabet,output="symbols",name="linear"):
        super().__init__()
        self._H = H
        self._alphabet = alphabet
        self.output = output
        self.name = name

    def set_H(self,H):
        """ set the channel matrix :math:`\mathbf{H}`."""
        self._H = H

    def linear_estimator(self,Y):
        raise NotImplementedError()

    def projector(self,X0):
        """ perform element-wise projection into the constellation """
        N_t, N = np.shape(X0)
        x_0 = np.atleast_2d(np.ravel(X0))
        error = np.abs(np.transpose(x_0) - self._alphabet)**2
        index = np.argmin(error,axis=1)
        X = index.reshape((N_t,N))
        return X.astype(int)

    def forward(self,Y):
        """ perform detection using the received samples :math:`\mathbf{Y}`."""
        X0 = self.linear_estimator(Y)
        X = self.projector(X0)

        if self.output != "symbols":
            X = self._alphabet[X]

        return X


class ZF_Detector(Linear_Detector):

    """Implements the Zero-Forcing (ZF) MIMO detector.

    Parameters
    ----------
    H : numpy array
        Channel matrix
    alphabet : numpy array
        symbol constellation 
    output : str, optional 
        specify if the forward function should output the symbols or the index.

    """

    def __init__(self,H,alphabet,output="symbols",name = "ZF"):
        super().__init__(H,alphabet,output=output,name = name)

    def linear_estimator(self,Y):
        """ perform linear estimation
        """
        H_inv = LA.pinv(self._H)
        X_est = np.matmul(H_inv,Y)
        return X_est


class MMSE_Detector(Linear_Detector):
    
    """Implements the MMSE MIMO detector.

    Parameters
    ----------
    H : numpy array
        Channel matrix
    sigma2: float
        noise variance
    alphabet : numpy array
        symbol constellation 
    output : str, optional 
        specify if the forward function should output the symbols or the index.

    """

    def __init__(self,H,sigma2,alphabet,output="symbols",name = "MMSE"):
        super().__init__(H,alphabet,output=output,name = name)
        self._sigma2 = sigma2

    def linear_estimator(self,Y):
        """ perform linear estimation.
        """
        N_r,N_t = self._H.shape
        H_H = np.conjugate(np.transpose(self._H))
        A = np.matmul(LA.inv(np.matmul(H_H,self._H)+self._sigma2*np.eye(N_t)),H_H)
        X_est = np.matmul(A,Y)
        return X_est
