import numpy as np
import pandas as pd
import configparser
from context import Sequential, Modulator, Recorder, Generator, Static_Channel, ZF_Detector, DL_Detector, get_alphabet, compute_ser
from scipy.stats import norm

# change backend for pandas (use plotly instead of matploltib)
pd.options.plotting.backend = "plotly"

# extract simulations parameters from config file
config = configparser.ConfigParser()
config.read('simulation2.ini')

modulation = config["input"]["modulation"]
order = int(config["input"]["order"])
N_r = int(config["channel"]["N_r"])
N_t = int(config["channel"]["N_t"])
N_trials = int(config["simulation"]["N_trials"])
SNR_min = float(config["simulation"]["SNR_min"])
SNR_max = float(config["simulation"]["SNR_max"])
SNR_step = float(config["simulation"]["SNR_step"])
N = int(config["training"]["N"])
SNR = float(config["training"]["SNR"])

# construct chain
H = norm.rvs(size=(N_r,N_t))+1j*norm.rvs(size=(N_r,N_t))
alphabet = get_alphabet(modulation,order)
generator = Generator(order)
recorder = Recorder()
modulator = Modulator(alphabet)
channel = Static_Channel(H)
model = Sequential([generator,recorder,modulator,channel])

# list of detectors
detector1 = ZF_Detector(H,alphabet)
detector2 = DL_Detector(N_r,N_t,alphabet)
detector_list = [detector1,detector2]
detector_names = ["ZF","DL"]

# training DL detector
print("(DL) Training Stage")
channel.set_SNR(SNR)
Y_training = model((N_t,N))
X_training = recorder.get_data()
detector2.train(Y_training,X_training,verbose=False,lr=10**-3)

# Monte Carlo Trials
print("Monte Carlo Trials")
SNR_vect = np.arange(SNR_min,SNR_max,SNR_step)
ser_data = np.zeros((len(SNR_vect),len(detector_list)))

for index_SNR in range(len(SNR_vect)):

    SNR = SNR_vect[index_SNR]
    print("SNR={}".format(SNR))
    channel.set_SNR(SNR)
    Y_test = model((N_t,N_trials))
    X_test = recorder.get_data()

    for index_detector in range(len(detector_list)):
        detector = detector_list[index_detector]
        X_estimated = detector(Y_test)
        error = compute_ser(X_estimated,X_test)
        print("{} : ser={}".format(detector.name,error))
        ser_data[index_SNR,index_detector] = error


# save and plot data using pandas + plotly
df = pd.DataFrame(data=ser_data, index=SNR_vect, columns=detector_names)
df.to_csv('./csv/simulation2_results.csv')
fig = df.plot()
fig.update_yaxes(type="log")
fig.show()