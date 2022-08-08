import numpy

class CTRNN:


    def __init__(self, size,weightRange=4,biasRange=2):
        self.size = size
        self.potentials = numpy.zeros(size)
        self.weights = (numpy.random.rand(size,size)-0.5)*0.01
        self.bias = (numpy.random.rand(size)-0.5)*0.01
        self.timescale = numpy.full(size,0.5)
        self.weightRange = weightRange
        self.biasRange = biasRange
        self.savedWeights = numpy.zeros((size,size))
        self.savedBias = numpy.zeros(size)
        self.savedTimescale = numpy.full(size,0.5)
        numpy.copyto(self.savedWeights,self.weights)
        numpy.copyto(self.savedBias,self.bias)
        self.weights += ((numpy.random.rand(size)-0.5)*0.01).clip(-1*weightRange,weightRange)
        self.bias += ((numpy.random.rand(size)-0.5)*0.01).clip(-1*biasRange,biasRange)
        self.timescale +=((numpy.random.rand(size)-0.5)*0.01).clip(0,1)
        


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
        return (1/(1+numpy.exp(-inputVector.clip(min=100))))

    def adapt(self,improvement,improvementThreshold=0.001,mutationSize=0.001):
        if(improvement>improvementThreshold):
            numpy.copyto(self.savedWeights,self.weights)
            numpy.copyto(self.savedBias,self.bias)
            numpy.copyto(self.savedTimescale,self.timescale)
            self.weights = (self.weights+(numpy.random.rand(self.size,self.size)-0.5)*mutationSize).clip(-1*self.weightRange,self.weightRange)
            self.bias = (self.bias+(numpy.random.rand(self.size)-0.5)*mutationSize).clip(-1*self.biasRange,self.biasRange)
            self.timescale = (self.timescale+(numpy.random.rand(self.size)-0.5)*mutationSize).clip(0,1)
        else:
            self.weights = (self.savedWeights + (numpy.random.rand(self.size,self.size)-0.5)*mutationSize).clip(-1*self.weightRange,self.weightRange)
            self.bias = (self.savedBias + (numpy.random.rand(self.size)-0.5)*mutationSize).clip(-1*self.biasRange,self.biasRange)
            self.timescale = (self.savedTimescale + (numpy.random.rand(self.size)-0.5)*mutationSize).clip(0,1)

    #single point crossover for weights, biases and timescale
    @staticmethod
    def recombine(brain1,brain2):

        #not comprehensive
        if(brain1.size != brain2.size or brain1.size<2):
            raise("brain mismatch")

        splitPoint = numpy.random.randint(1,brain1.size)
        newWeights = numpy.concatenate((brain1.weights[:splitPoint],brain2.weights[splitPoint:]))
        newBias = numpy.concatenate((brain1.bias[:splitPoint],brain2.bias[splitPoint:]))
        newTimescale = numpy.concatenate((brain1.timescale[:splitPoint],brain2.timescale[splitPoint:]))

        newBrain = CTRNN(brain1.size)
        newBrain.weights=newWeights
        newBrain.savedWeights=newWeights
        newBrain.bias=newBias
        newBrain.savedBias=newBias
        newBrain.timescale=newTimescale
        newBrain.savedTimescale=newTimescale
        return newBrain

    def mutate(self,mutationSize=0.01):
        self.weights = (self.weights+(numpy.random.rand(self.size,self.size)-0.5)*mutationSize).clip(-1*self.weightRange,self.weightRange)
        self.bias = (self.bias+(numpy.random.rand(self.size)-0.5)*mutationSize).clip(-1*self.biasRange,self.biasRange)
        self.timescale = (self.timescale+(numpy.random.rand(self.size)-0.5)*mutationSize).clip(0,1)
        numpy.copyto(self.savedWeights,self.weights)
        numpy.copyto(self.savedBias,self.bias)
        numpy.copyto(self.savedTimescale,self.timescale)
        





        
        
        





