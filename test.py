from cgi import test
from doctest import OutputChecker
from lib.PyCTRNN import CTRNN
import numpy as np

brain = CTRNN(4)
brain.weights = np.zeros((4,4))
brain.weights[1,0] = -1
inputVec = np.array([1,0,0,0])

# print(brain.weights)

testNum = 1
print(1/(1+np.exp(-testNum)))
print(brain.sigmoid(np.array([testNum])))

# for i in range(4):
#     out = brain.step(inputVec)
#     # print(out)

