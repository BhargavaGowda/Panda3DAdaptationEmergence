import matplotlib.pyplot as pl
import numpy as np

data1Label = "BasicFeedForwardMask"
data2Label = "BasicMaskMut4"
data3Label = "Test_BasicMask"
data4Label = "Discrete20SameTimeFit3Trend"


data1 = np.loadtxt("results/" + data1Label + ".csv")
data2 = np.loadtxt("results/" + data2Label + ".csv")
# data3 = np.loadtxt("results/" + data3Label + ".csv")
# data4 = np.loadtxt("results/" + data4Label + ".csv")

data = [data1,data2]

fig1, ax1 = pl.subplots()
ax1.set_title('Fitness')
ax1.boxplot(data,labels=[data1Label,data2Label])


# pl.figure(figsize=(7.5, 5))
# pl.plot(data1,label=data1Label)
# pl.plot(data2,label=data2Label,color="green")
# pl.plot(data3,label=data3Label,color="red")
# # pl.plot(data4,label=data4Label,color="green")
pl.legend()
pl.show()