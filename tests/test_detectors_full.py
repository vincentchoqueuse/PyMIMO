import numpy as np
import pandas as pd
import configparser
from context import Sequential, Modulator, Recorder, Generator, Static_Channel, ML_Detector, DL_Detector, ZF_Detector, MMSE_Detector, get_alphabet, compute_ser
from scipy.stats import norm

# change backend for pandas (use plotly instead of matploltib)
pd.options.plotting.backend = "plotly"

# extract simulations parameters from config file
config = configparser.ConfigParser()
config.read('simulation1.ini')

modulation = config["input"]["modulation"]
order = int(config["input"]["order"])
N_r = 5
N_t = 3
N_trials = int(config["simulation"]["N_trials"])
SNR_min = -6
SNR_max = 15
SNR_step = 1

# construct chain
H = norm.rvs(size=(N_r,N_t))+1j*norm.rvs(size=(N_r,N_t))
alphabet = get_alphabet(modulation,order,type="gray",norm=True)
generator = Generator(order)
recorder = Recorder()
modulator = Modulator(alphabet)
channel = Static_Channel(H)
model = Sequential([generator,recorder,modulator,channel])

# Monte Carlo Trials
print("Monte Carlo Trials")
detector1 = ML_Detector(H,alphabet)
detector2 = ZF_Detector(H,alphabet)
detector3 = MMSE_Detector(H,0,alphabet)
detector4 = DL_Detector(N_r,N_t,alphabet)
detector_list = [detector1, detector2, detector3, detector4]
detector_names = ["ML","ZF","MMSE","DL"]
Nb_detectors = len(detector_names)
SNR_vect = np.arange(SNR_min,SNR_max,SNR_step)

# training DL detector
print("(DL) Training Stage")
channel.set_SNR(4)
Y_training = model((N_t,50000))
X_training = recorder.get_data()
detector4.train(Y_training,X_training,verbose=False,lr=10**-3)

ser_list = []
ser_data = np.zeros((len(SNR_vect),Nb_detectors))
N_trials = 50000

for index_SNR in range(len(SNR_vect)): # main loop (loop over SNR)

    # change SNR
    SNR = SNR_vect[index_SNR]
    print("SNR={}".format(SNR))
    channel.set_SNR(SNR)

    #ser value
    ser_list = np.zeros((N_trials,Nb_detectors))

    for trial in range(N_trials):
        Y_test = model((N_t,1))         # create data of size N_t*1
        X_test = recorder.get_data()    # extract the generated input data (after the generator)
        sigma2 = channel._sigma2   # extact the noise variance

        for indice in range(Nb_detectors):  # loop over detectors

            detector = detector_list[indice]
            if indice == 2: 
                detector._sigma2 = sigma2 # for MMSE detection

            X_estimated = detector(Y_test)
            x_estimated = np.ravel(X_estimated)
            x_test = np.ravel(X_test)
            error = compute_ser(x_estimated,x_test)
            ser_list[trial,indice] = error

    ser_value = np.mean(ser_list,axis=0)
    ser_data[index_SNR,:] = ser_value
    print("ser={}".format(ser_value))
    

# save and plot data using pandas + plotly
df = pd.DataFrame(data=ser_data, index=SNR_vect, columns=detector_names)
df.to_csv('./csv/simulation0_results.csv')
fig = df.plot()
fig.update_yaxes(type="log")
fig.show()