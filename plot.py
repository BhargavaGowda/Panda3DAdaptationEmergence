import matplotlib.pyplot as pl
import numpy as np

data1 = np.loadtxt("results/Mut01Trend.csv")
data2 = np.loadtxt("results/Mut1Trend.csv")
data3 = np.loadtxt("results/MutInterleavedTrend.csv")
data4 = np.loadtxt("results/MutInterleaved2xTrend.csv")

pl.plot(data1)
pl.plot(data2)
pl.plot(data3)
pl.plot(data4)
pl.show()