import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

current_dir = os.path.abspath(os.path.dirname(__file__))

chkpt_dir = current_dir +  '/data_reward'


check_data_reward = pd.read_csv(chkpt_dir)
r = check_data_reward['reward']
r = r[8000:]
def average_every_n(lst, n):
    return [sum(lst[i:i+n])/n for i in range(0, len(lst), n)]

r = average_every_n(r, 10)

window_size = 20

mean_list = []
ci_lower_list = []
ci_upper_list = []


for i in range(len(r)):
    if i < window_size:
        window = r[:i+1]
    else:
        window = r[i-window_size:i]
    mean = np.mean(window)
    ci_lower = np.min(window)
    ci_upper = np.max(window)
    
    mean_list.append(mean)
    ci_lower_list.append(ci_lower)
    ci_upper_list.append(ci_upper)

mean_list = mean_list[:-window_size] + [np.nan]*window_size
ci_lower_list = ci_lower_list[:-window_size] + [np.nan]*window_size
ci_upper_list = ci_upper_list[:-window_size] + [np.nan]*window_size

plt.figure(figsize=(6.5, 5))

plt.grid(linestyle='-.')
plt.xticks(size=14)
plt.yticks(size=14)
plt.xlabel('Training steps * 10', fontsize=14)
plt.ylabel('Reward', fontsize=14)


plt.fill_between(range(len(r)), ci_lower_list, ci_upper_list, color='gray', alpha=0.4)


plt.plot(range(len(r)), mean_list, color='black', linewidth=2)

plt.savefig('training_4_6.png',dpi=300, bbox_inches = 'tight')


plt.show()

