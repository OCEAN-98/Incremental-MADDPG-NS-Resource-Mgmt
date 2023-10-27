import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import stats

current_dir = os.path.abspath(os.path.dirname(__file__))

chkpt_dir = current_dir +  '/data_reward'

check_data_reward = pd.read_csv(chkpt_dir)
r = check_data_reward['reward']
r = r[8000:16000]
def average_every_n(lst, n):
    return [sum(lst[i:i+n])/n for i in range(0, len(lst), n)]

r = average_every_n(r, 10)
window_size = 20

confidence_level = 0.95

mean_list = []
ci_lower_list = []
ci_upper_list = []

for i in range(len(r)):
    if i < window_size:
        window = r[:i+1]
    else:
        window = r[i-window_size:i]
    mean = np.mean(window)
    n = len(window)
    bootstrap_means = []
    for _ in range(1000): 
        resampled_data = np.random.choice(window, size=n, replace=True)
        bootstrap_mean = np.mean(resampled_data)
        bootstrap_means.append(bootstrap_mean)
    bootstrap_means = np.array(bootstrap_means)
    ci_lower = np.percentile(bootstrap_means, (1 - confidence_level) / 2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 + confidence_level) / 2 * 100)

    mean_list.append(mean)
    ci_lower_list.append(ci_lower)
    ci_upper_list.append(ci_upper)

plt.figure(figsize=(6.5, 5))


plt.grid(linestyle='-.')
plt.xticks(size=14)
plt.yticks(size=14)
plt.xlabel('Training steps * 10', fontsize=14)
plt.ylabel('Reward', fontsize=14)


plt.fill_between(range(len(r)), ci_lower_list, ci_upper_list, color='gray', alpha=0.4)


plt.plot(range(len(r)), mean_list, color='black', linewidth=2)

plt.savefig('training_4.png',dpi=300, bbox_inches = 'tight')

plt.show()

