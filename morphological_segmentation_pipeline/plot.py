import matplotlib.pyplot as plt
import numpy as np 

x = range(100)
y = range(100,200)
fig = plt.figure()
ax = fig.add_subplot(111)

markers=[',', '+', '_', '.', 'o', '*', 'p', 's', 'x', 'h', '^', 'D']
marker_count = 0
subject = [3,4,5,6,7,9,10,11,12,13,14,15]
subject_tp=[3,3,3,5,5,3,1,1,3,3,3,3]
color = ['b', 'g', 'r', 'c', 'm', 'y', "peachpuff", "fuchsia", [0.2,0.2,0.6],[0.3,0.7,0.4],[0.6,0.8,0.1],[0.5,0.2,0.7]]

from xlrd import open_workbook
wb = open_workbook('/Volumes/Studies/LEGmuscle/Result/legVolume_2Jan2018.xlsx')
for s in wb.sheets():
    #print 'Sheet:',s.name
    values = []
    for row in range(s.nrows):
        col_value = []
        for col in range(s.ncols):
            value  = (s.cell(row,col).value)
            try : value = str(double(value))
            except : pass
            col_value.append(value)
        values.append(col_value)
data = np.array(values)
print(data.shape)

# data = 

# tp = [1,2,3]
s_count = 0
row_count = 1
for i in subject:
# for i in range(1):
    row_count += 1
    print(row_count)
    bl_left = data[row_count:row_count+subject_tp[s_count],4].astype(float)
    bl_right = data[row_count:row_count+subject_tp[s_count],11].astype(float)
    # print('npshape',(np.arange(subject_tp[s_count]).shape))
    # print(bl_left.shape)
    # ax.scatter(np.arange(subject_tp[s_count]), bl_left, s=10, c='b', marker=markers[marker_count])
    tp = np.arange(subject_tp[s_count])
    tp = np.add(tp,1)
    print(bl_left)
    print(bl_right)
    ax.plot(tp, bl_left,c=color[s_count],label=subject[s_count],linestyle="solid", marker="o")
    # ax.scatter(np.arange(subject_tp[s_count]), bl_right, s=10, c='b', marker=markers[marker_count])
    ax.plot(tp, bl_right,c=color[s_count],linestyle="solid", marker="o")
    marker_count += 1
    row_count = row_count + subject_tp[s_count]
    s_count += 1
    print(row_count)
x = [1, 2, 3, 4]
ax.set_title('Leg Muscle Volume Change by Subject')
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
ax.set_ylabel('Leg Muscle Volume')
ax.set_xlabel('Time Point')
ax.set_xticklabels(('BL','FU','FU2','FU3'))
# ax.set_yticklabels([])

# # ax1.scatter([3,3], [138775.1300,143844.0417], s=10, c='b', marker="s", label='BL')
# # ax1.scatter(x[40:],y[40:], s=10, c='r', marker="o", label='second')
plt.legend(loc='upper left');
plt.show()