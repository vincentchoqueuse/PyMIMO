import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..core import Processor

class FullyConNet(nn.Module):

    def __init__(self,N_in,N_out,N_neurons):
        super(FullyConNet, self).__init__()
        self.fc1 = nn.Linear(N_in, N_neurons) 
        self.fc2 = nn.Linear(N_neurons,N_neurons)
        self.fc3 = nn.Linear(N_neurons, N_neurons)
        self.fc4 = nn.Linear(N_neurons, N_neurons)
        self.fc5 = nn.Linear(N_neurons, N_neurons)
        self.fc6 = nn.Linear(N_neurons, N_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class DL_Detector(Processor):

    """DL Detector 

    This class performs Deep Learning-based detection in MIMO systems. The DL network is a feedforward network using 5 hidden Dense layers with reLU activation function.
    The network output is a one-hot encoding vector :math:`\mathbf{u}` of size :math:`N_tM\\times 1` where

    .. math ::

        \\mathbf{u}=\\left[\\begin{array}{c}u_{1}[0]\\\\ \\vdots \\\\ u_{M}[0] \\\\ \\vdots \\\\ u_1[N_t]\\\\ \\vdots \\\\ u_M[N_t]   \\end{array}\\right]


    Parameters
    ----------
    N_in : int
        number of received samples
    N_out : int
        number of transmitted symbols
    alphabet : numpy array
        symbol constellation 
    N_neurons : int, optional
        number of neurons in each hidden layer
    output : str, optional 
        specify if the forward function should output the symbols or the index.


    """

    def __init__(self,N_in,N_out,alphabet,N_neurons=100,output = "symbols",name="dl_detector"):
        super().__init__()
        self._fullycon = FullyConNet(2*N_in,len(alphabet)*N_out,N_neurons)
        self._alphabet = alphabet
        self.output = output
        self.name = name

    def one_hot_encoding(self,X):
        N_alphabet = len(self._alphabet)
        N1, N_ex = X.shape
        Y = np.zeros((N_alphabet*N1,N_ex))

        for m in range(N_ex):
            for n in range(N1):
                index = int(X[n,m])
                Y[n*N_alphabet+index,m] = 1 

        return torch.from_numpy(np.transpose(Y).astype(np.float32))

    def transform_input(self,X):
        X = np.vstack([np.real(X),np.imag(X)])
        return torch.from_numpy(np.transpose(X).astype(np.float32))

    def transform_output(self,Y,reduction="max"):
        N_batch, N = Y.shape
        N_alphabet = len(self._alphabet)
        Y = torch.reshape(Y, (N_batch, int(N/N_alphabet),N_alphabet))
        if reduction == "max":
            Y = torch.argmax(Y,dim=2)

        return np.transpose(Y.detach().numpy())

    def train(self,X,Y,num_epoch=100,batch_size=1000, lr=10**-3,verbose=True):
        N = X.shape[1]
        X_input = self.transform_input(X) 
        Y_target = self.one_hot_encoding(Y)
        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(self._fullycon.parameters(),lr=lr)

        for epoch in range(num_epoch):  # loop over the dataset multiple times

            running_loss = 0.0

            for i in range(0,N,batch_size):
                
                inputs = X_input[i:i+batch_size,:]
                labels = Y_target[i:i+batch_size,:]

                optimizer.zero_grad()
                outputs = self._fullycon(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # print statistics
            if verbose == False:
                print('epoch={} : loss={}'.format(epoch,running_loss))

    def forward(self,X):
        
        X = self.transform_input(X)

        with torch.no_grad(): # disable gradient computation
            X = self._fullycon(X)

        X = self.transform_output(X)

        if self.output != "symbols":
            X = self._alphabet[X]

        return X
