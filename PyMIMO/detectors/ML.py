import numpy as np
import numpy.linalg as LA
import itertools
import numpy as np
from ..core import Processor

class ML_Detector(Processor):

    """Implements the ML Detector for white Gaussian noise.

    Parameters
    ----------
    H : numpy array
        Channel matrix
    alphabet : numpy array
        symbol constellation 
    output : str, optional 
        specify if the forward function should output the symbols or the index.


    """

    def __init__(self,H,alphabet,output="symbols",name = "ML"):
        super().__init__()
        self._H = H
        self._alphabet = alphabet
        self.output = output
        self.name = name

    def set_H(self,H):
        """ set the channel matrix :math:`\mathbf{H}`."""
        self._H = H

    def get_nb_candidates(self):
        N_r, N_t = self._H.shape
        alphabet = self._alphabet
        return len(alphabet)**N_t

    def get_candidates(self,alphabet,N_t):
        symbols = np.arange(len(alphabet))
        input_list = [p for p in itertools.product(symbols, repeat=N_t)]

        # preallocation of memory
        X = np.zeros((N_t,len(input_list)),dtype=np.complex)
        S = np.zeros((N_t,len(input_list)))

        for indice in range(len(input_list)):
            input = np.array(input_list[indice])  # store combinaison
            x = self._alphabet[input]             # transmitted data
            X[:,indice] = x
            S[:,indice] = input

        return X,S

    def forward(self,Y):
        """ performs detection using the received samples :math:`\mathbf{Y}`."""
        N_r, N_t = self._H.shape
        N_r, N = Y.shape
        X = np.zeros((N_t,N),dtype=int)
        alphabet = self._alphabet
        
        X_candidates, S_candidates = self.get_candidates(alphabet,N_t)
        
        Y_candidates = np.matmul(self._H,X_candidates)  # compute all combinaison of received data

        for n in range(N):
            y = np.transpose(np.atleast_2d(Y[:,n]))
            index_min = np.argmin(np.sum(np.abs(y-Y_candidates)**2,axis=0))
            X[:,n] = S_candidates[:,index_min]

        if self.output != "symbols":
            X = self._alphabet[X]

        return X
