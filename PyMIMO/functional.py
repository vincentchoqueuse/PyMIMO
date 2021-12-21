import numpy as np
import importlib_resources as pkg_resources
import os.path as path

def get_alphabet(modulation,order,type="gray",norm=True):
    # extract alphabet
    pathname = path.dirname(path.abspath(__file__))
    filename = "{}/data/{}_{}_{}.csv".format(pathname,modulation,order,type)
    data = np.loadtxt(filename,delimiter=',',skiprows=1)
    alphabet = data[:,1]+1j*data[:,2]

    if norm == True :
        alphabet = alphabet/np.sqrt(np.mean(np.abs(alphabet)**2))

    return alphabet

def sym_2_bin(sym,width=4):

    data = []
    for indice in range(len(sym)):
        data.append(np.binary_repr(sym[indice],width))

    string = ''.join(data)

    return np.array(list(string), dtype=int)

def compute_ser(X_target,X_detected):
    nb_errors = np.count_nonzero(X_target-X_detected)
    return nb_errors / X_detected.size


