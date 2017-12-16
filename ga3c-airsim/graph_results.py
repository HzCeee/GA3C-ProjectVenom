import matplotlib.pyplot as plt
import statistics
#log_file = open('./move_by_v.txt', 'r')
#log_file = open('./moving_target_no_early_stop_4f+d.txt', 'r')
log_file = open('./phys.txt','r')
log_file = open('./phys_bigger_queue.txt','r')
line = log_file.readline()
x = []
y = []
average = []
median = []
t = 1000.0
last_t = []
i = 0
avg_len = 0
while line:
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
        x.append(ep_nb)
        last_t.append(rew)
        average.append(sum(last_t)/len(last_t))
    elif ep_len > 1:
        y.append(rew)
        x.append(ep_nb)
        last_t.pop(0)
        last_t.append(rew)
        average.append(sum(last_t)/t)
#        median.append(statistics.median(last_t))
    #except:
    #    print
    line = log_file.readline()
fig, ax = plt.subplots()
print('Last 1000 Reward: '+str(sum(last_t)/t))
print('Avg Len: '+str(float(avg_len)/float(i)))
#ax.plot(x, y)
ax.plot(x,average)
#ax.plot(x, median)
ax.set(xlabel='time', ylabel='dist moved to target')
ax.grid()
plt.show()
