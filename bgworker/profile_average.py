import numpy as np
import matplotlib.pyplot as plt

fnames = ['45_85_85', '95_70_35', '110_100_85', '140_110_125', '165_105_65']

for f in fnames:
    x_fnames = ['qian30us/x_%s_%d' % (f, i) for i in range(1, 6)]
    y_fnames = ['qian30us/y_%s_%d' % (f, i) for i in range(1, 6)]
    xs = []
    ys = []
    for x_file, y_file in zip(x_fnames, y_fnames):
        x_data = np.loadtxt(x_file, skiprows=2)
        y_data = np.loadtxt(y_file, skiprows=2)
        xs.append(x_data)
        ys.append(y_data)
    xs = np.array(xs)
    ys = np.array(ys)
    x_data = np.median(xs, axis=0)
    y_data = np.median(ys, axis=0)[::-1]
    np.savetxt('profile-results/x_%s' % f, np.c_[np.zeros(x_data.shape[0]), x_data])
    np.savetxt('profile-results/y_%s' % f, np.c_[np.zeros(y_data.shape[0]), y_data])


        
