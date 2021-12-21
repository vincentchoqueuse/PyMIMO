import numpy as np 
import matplotlib.pyplot as plt
from ..core import Processor

class Scope(Processor):

    def __init__(self,figure=0):
        super().__init__()
        self._figure = figure

    def forward(self,X):

        plt.figure(self._figure)
        plt.plot(np.real(X),np.imag(X),'b*')

        return X

class Recorder(Processor):

    def __init__(self):
        super().__init__()
        self._data = None

    def get_data(self):
        return self._data

    def forward(self,X):
        self._data = X
        return X