import matplotlib.pyplot as pl
import numpy as np

data1 = np.loadtxt("results/FullMaskTrend.csv")
# data2 = np.loadtxt("results/2stgMaskTrendPart2.csv")
data3 = np.loadtxt("results/BasicMaskTrend.csv")
# data4 = np.loadtxt("results/FullMaskTrend.csv")

pl.plot(data1)
# pl.plot(data2)
pl.plot(data3)
# pl.plot(data4)
pl.show()