import numpy

class CTRNN:


    def __init__(self, size):
        self.size = size
        self.potentials = numpy.zeros(size)
        self.weights = (numpy.random.rand(size,size)-0.5)*0.01
        self.bias = (numpy.random.rand(size)-0.5)*0.01
        self.timescale = numpy.full(size,0.5)
        self.savedWeights = numpy.zeros((size,size))
        self.savedBias = numpy.zeros(size)
        numpy.copyto(self.savedWeights,self.weights)
        numpy.copyto(self.savedBias,self.bias)
        self.weights += (numpy.random.rand(size,size)-0.5)*0.001
        self.bias += (numpy.random.rand(size)-0.5)*0.001
        self.radiation = 1
        


    def getWeights(self):
        return self.weights

    def getBias(self):
        return self.bias

    def getTimescale(self):
        return self.timescale
    
    def getPotentials(self):
        return self.potentials

    def step(self, inputArray):
        self.potentials += self.timescale*(inputArray - self.potentials + numpy.matmul(self.weights, self.sigmoid(self.potentials+self.bias)))
        return self.potentials

    def sigmoid(self,inputVector):
        return (1/(1+numpy.exp(-inputVector)))


    def adapt(self,improvement,improvementThreshold=0,mutationSize=0.001,radiationStrength=1.5):
        if(improvement>improvementThreshold):
            # self.radiation = 1
            numpy.copyto(self.savedWeights,self.weights)
            numpy.copyto(self.savedBias,self.bias)
            self.weights += (numpy.random.rand(self.size,self.size)-0.5)*mutationSize
            self.bias += (numpy.random.rand(self.size)-0.5)*mutationSize
        else:
            self.weights = self.savedWeights + (numpy.random.rand(self.size,self.size)-0.5)*mutationSize#*self.radiation
            self.bias = self.savedBias + (numpy.random.rand(self.size)-0.5)*mutationSize#*self.radiation
            # self.radiation*=radiationStrength
            # self.radiation = min(self.radiation,1000)
        print("--")      
        print(improvement)
        # print(self.radiation)






    # def adapt(self,feedback,weightMut=3,biasMut=3,timeMut=0):
    #     feedback = max(0,min(feedback,1))
    #     #where to go
    #     self.weightsTrajectory = self.weightsTrajectory*feedback + (1-feedback)*weightMut*(numpy.random.rand(self.size,self.size)-0.5)
    #     self.biasTrajectory = self.biasTrajectory*feedback + (1-feedback)*biasMut*(numpy.random.rand(self.size)-0.5)
    #     self.timescaleTrajectory = self.timescaleTrajectory*feedback + (1-feedback)*timeMut*(numpy.random.rand(self.size)-0.5)
    #     #how quick to go there
    #     self.weights += self.weightsTrajectory*self.weightLearnRate
    #     self.bias += self.biasTrajectory*self.biasLearnRate
    #     self.timescale += self.timescaleTrajectory*self.timescaleLearnRate
    #     # print(self.weightsTrajectory)
    #     # print(self.weights)
    #     # print(self.biasTrajectory)
    #     # print(self.bias)
    #     # print(self.timescale)

