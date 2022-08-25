import matplotlib.pyplot as pl
import numpy as np

data1 = np.loadtxt("results/TConstTimeTrend.csv")
data2 = np.loadtxt("results/TConstTime01Trend.csv")
data3 = np.loadtxt("results/TConstTimeAnnealTrend.csv")
# data4 = np.loadtxt("results/TConstTimeTrend.csv")

pl.plot(data1)
pl.plot(data2)
pl.plot(data3)
# pl.plot(data4)
pl.show()