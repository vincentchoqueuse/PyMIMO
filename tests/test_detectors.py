import numpy as np
import pandas as pd
import configparser
from context import Sequential, Modulator, Recorder, Generator, Gaussian_Channel, ML_Detector, ZF_Detector, MMSE_Detector, get_alphabet, compute_ser

# change backend for pandas (use plotly instead of matploltib)
pd.options.plotting.backend = "plotly"

# extract simulations parameters from config file
config = configparser.ConfigParser()
config.read('simulation1.ini')

modulation = config["input"]["modulation"]
order = int(config["input"]["order"])
N_r = int(config["channel"]["N_r"])
N_t = int(config["channel"]["N_t"])
N_trials = int(config["simulation"]["N_trials"])
SNR_min = float(config["simulation"]["SNR_min"])
SNR_max = float(config["simulation"]["SNR_max"])
SNR_step = float(config["simulation"]["SNR_step"])

# construct chain
alphabet = get_alphabet(modulation,order,type="gray",norm=True)
generator = Generator(order)
recorder = Recorder()
modulator = Modulator(alphabet)
channel = Gaussian_Channel(N_r)
model = Sequential([generator,recorder,modulator,channel])

# Monte Carlo Trials
print("Monte Carlo Trials")
detector_names = ["ML","ZF","MMSE"]
Nb_detectors = len(detector_names)
SNR_vect = np.arange(SNR_min,SNR_max,SNR_step)

ser_list = []
ser_data = np.zeros((len(SNR_vect),Nb_detectors))

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
        H = channel.get_H()             # retrieve channel parameters (H, sigma2)
        sigma2 = channel.get_sigma2()   # extact the noise variance

        for indice in range(Nb_detectors):  # loop over detectors

            if indice == 0:
                detector = ML_Detector(H,alphabet)
            if indice == 1:
                detector = ZF_Detector(H,alphabet)
            if indice == 2: 
                detector = MMSE_Detector(H,sigma2,alphabet)

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
df.to_csv('./csv/simulation1_results.csv')
fig = df.plot()
fig.update_yaxes(type="log")
fig.show()