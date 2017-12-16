import matplotlib.pyplot as plt
import statistics
import numpy as np
def num_greater(l, n):
    nb = 0
    for i in range(len(l)):
        if l[i] > n: nb += 1
    return nb

#log_file = open('./move_by_v_no_c.txt', 'r')
#log_file = open('./moving_target_4f+d.txt', 'r')
#log_file = open('./moving_target_no_early_stop_4f+d.txt', 'r')
#log_file = open('./phys.txt','r')
#log_file = open('./phys_bigger_queue.txt','r')
#log_file = open('./phys_bigger_queue_bigger_network.txt','r')
log_file = open('./deep_wide_phys.txt')
line = log_file.readline()
x = []
y = []
average = []
success = []
success_rate = []
median = []
t = 2000.0
last_t = []
i = 0
avg_len = 0
ones = np.array([1]*int(t))
while line and i < 150000:
    while '[' not in line:
        line = log_file.readline()
    sections = line.split('[')
    info = sections[2][:-2].split()
    
    ep_nb = int(info[1])
    ep_len = int(info[3])
    avg_len += ep_len
    i += 1
    rew = float(info[5])

    if len(last_t) < t and ep_len > 1:
        #garbage=1
        last_t.append(rew)
        if rew > 1000: success.append(1)
        else: success.append(0)
        #average.append(sum(last_t)/len(last_t))
    elif ep_len > 1:
        x.append(ep_nb)
        y.append(rew)
        last_t.pop(0)
        last_t.append(rew)
        success.pop(0)
        if rew > 101: success.append(1)
        else: success.append(0)
        success_rate.append(100.0*sum(success)/t)
        average.append(sum(last_t)/t)
#        median.append(statistics.median(last_t))
    #except:
    #    print
    line = log_file.readline()
fig, ax = plt.subplots()
print('Last 1000 Reward: '+str(sum(last_t)/t))
print('Avg Len: '+str(float(avg_len)/float(i)))
#ax.plot(x, y)
#ax.plot(x,average)
ax.plot(x,success_rate)
#ax.plot(x, median)
ax.set(xlabel='time', ylabel='% successful runs')
ax.grid()
plt.show()
