## Best is to fill that up as you do the measurements
import numpy as np
import sys

delay_first_meas = 30 # sec
time_between_meas = 17 # sec

acqdec = input('Acquisition or decay? (a/d)  ')
if acqdec == 'a':
       suffix = '_VRMACQtime'
elif acqdec == 'd':
       suffix = '_VRMDECtime'

nb_cons = input('Number of consecutive measurements? (default = 1)  ')
if nb_cons == '':
       nb_cons = 1
else:
       nb_cons = int(eval(nb_cons))
print('For the first measurement of the series:')
day = input(' * Number of days since last measurement? (default = 0)  ')
if day =='': day = 0
else: day = int(eval(day))
hour = input(' * Number of hours since last measurement? (default = 0)  ')
if hour =='': hour = 0
else: hour = int(eval(hour))
min = input(' * Number of minutes since last measurement? (default = 0)  ')
if min =='': min = 0
else: min = int(eval(min))
sec = input(' * Number of seconds since last measurement? (default = 0)  ')
if sec =='': sec = 0
else: sec = int(eval(sec))

if len(sys.argv) == 2:
       time = []
       fp = open(sys.argv[1],'r')
       for j,line in enumerate(fp):
              cols = line.split()
              time.append(int(cols[0]))
       fp.close()
       lasttime = time[-1]
elif len(sys.argv) == 1:
       lasttime = delay_first_meas

newtime = lasttime + (day*24+hour)*3600 + min*60 + sec

if len(sys.argv) == 2:
       fp = open(sys.argv[1], 'a')
       for k in np.arange(nb_cons):
              newtime += time_between_meas
              fp.write(str(newtime)+'\n')
       fp.close()
elif len(sys.argv) == 1:
       name = input('Name of file? (include path)  ')
       fp = open(name+suffix+'.txt', 'w')
       for k in np.arange(nb_cons):
              newtime += time_between_meas
              fp.write(str(newtime) + '\n')
       fp.close()