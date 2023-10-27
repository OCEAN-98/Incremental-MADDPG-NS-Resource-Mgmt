import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.interpolate import splrep, splev

def average_every_n(lst, n):

    return [np.mean(lst[i:i+n]) for i in range(0, len(lst), n)]

current_dir = os.path.abspath(os.path.dirname(__file__))
chkpt_dir = current_dir + '/data_reward'
check_data_reward = pd.read_csv(chkpt_dir)
reward = check_data_reward['reward']

window_size = 50

averages = average_every_n(reward, window_size)
average_std = np.std(averages)

# Create the B-spline representation of the curve

tck_avg = splrep(range(len(averages)), averages, s=0.01)  # reduced the smoothing factor
tck_lower = splrep(range(len(averages)), [avg - average_std for avg in averages], s=0.01)
tck_upper = splrep(range(len(averages)), [avg + average_std for avg in averages], s=0.01)
# Generate new smoothed curves

xnew = np.linspace(0, len(averages) - 1, num=1000)
ynew_avg = splev(xnew, tck_avg)
ynew_lower = splev(xnew, tck_lower)
ynew_upper = splev(xnew, tck_upper)

plt.rcParams["font.weight"] = "bold"
plt.grid(linestyle='-.')
plt.xticks(size=16, fontweight='bold')
plt.yticks(size=16, fontweight='bold')
plt.xlabel('Training steps', fontsize=16, fontweight='bold')
plt.ylabel('Reward', fontsize=16, fontweight='bold')

plt.plot(xnew, ynew_avg, color='blue')  # the mean curve.
plt.fill_between(xnew, ynew_lower, ynew_upper, color='blue', alpha=0.2)  # the std band.
plt.show()