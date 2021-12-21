from scipy.stats import randint
from ..core import Processor

class Sequential(Processor):

    def __init__(self,processor_list,name="sequential"):
        self.name = name
        self._processor_list = processor_list

    def forward(self,X):
        Y = X
        for processor in self._processor_list:
            Y = processor(Y)
        return Y

class Generator(Processor):

    def __init__(self,L):
        super().__init__()
        self._L = L

    def forward(self,size):
        X = randint.rvs(0, self._L, size=size) 
        return X

class Modulator(Processor):

    def __init__(self,alphabet=True):
        super().__init__()
        self._alphabet = alphabet

    def get_alphabet(self):
        return self._alphabet

    def forward(self,X):
        Y = self._alphabet[X]
        return Y