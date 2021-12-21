import numpy as np
import numpy.linalg as LA
import itertools
import numpy as np
from ..core import Processor

class ML_Detector(Processor):

    """ML Detector 

    This class performs ML detection in MIMO systems
    
    .. math :: 

        \widehat{\mathbf{x}} = \\arg \min_{\mathbf{x}\in \mathcal{M}^{N_t}}\|\mathbf{y}-\mathbf{H}\mathbf{x}\|^2
    
    where :math:`\mathcal{M}` is the constellation of the transmitted data, and :math:`\mathbf{H}` is the channel matrix.

    """

    def __init__(self,H,alphabet,output="symbols",name = "ML"):
        super().__init__()
        self._H = H
        self._alphabet = alphabet
        self.output = output
        self.name = name

    def set_H(self,H):
        """ set the channel matrix :math:`H`."""
        self._H = H

    def get_nb_candidates(self):
        N_r, N_t = self._H.shape
        alphabet = self._alphabet
        return len(alphabet)**N_t

    def get_candidates(self,alphabet,N_t):
        """ get all combinaisons of N_t transmitted data belonging to a particular constellation."""
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
        """ perform detection using the received samples :math:`\mathbf{Y}`."""
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
