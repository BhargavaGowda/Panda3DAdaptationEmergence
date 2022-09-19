from lib.PyCTRNN import CTRNN
import random
from lib.SimBase import SimBase
import sys
import numpy as np
from panda3d.core import Vec3,DirectionalLight


class paddleBot(SimBase):

    def __init__(self, fStartDirect=True, windowType=None):
        super().__init__(fStartDirect, windowType)

        # Sim Parameters
        self.realTime = False
        self.maxJointAngle = 80
        self.jointSpeed = 10
        self.ballSpeed = 5
        self.brainSize = 10
        self.numDrops = 20
        self.paddleSize = 0.5
        self.maxGen = 200
        self.mut = 1
        self.crossRate = 0.5
        self.numTrials = 30
        self.saveBrain = True
        self.retrieveBrain = False
        self.maxPop = 10

        mask = np.ones((self.brainSize,self.brainSize))
        # mask[-3,:6] = 1
        mask[-2,:6] = 1
        mask[-1,:6] = 1

        # init
        self.currentScore = 0
        self.bestBrainScore = 0
        self.ballCounter = 0
        self.gen = 0
        self.currentTrial = 0
        self.individualIndex = 0
        self.retest = False
        self.pop = []

        for i in range(self.maxPop):
            newBrain = CTRNN(self.brainSize)
            newBrain.mask = mask
            newBrain.applyMask()
            self.pop.append([newBrain,0])
        self.brain = self.pop[self.individualIndex][0]
        self.bestBrain = self.brain


        
        
        

        # Logging
        self.saveName = "Paddle4"
        self.loadName = "Paddle4"
        self.fitnessTrend = np.zeros(self.numTrials)
        if(self.retrieveBrain == True):
            print("Loading Brain:",self.loadName)
            self.brain.weights = np.loadtxt("results/brains/Size_" + str(self.brain.size)  + "_" + self.loadName + "_Weights.csv",delimiter=",")
            self.brain.bias = np.loadtxt("results/brains/Size_" + str(self.brain.size)  + "_" + self.loadName + "_Bias.csv",delimiter=",")
            self.brain.timescale = np.loadtxt("results/brains/Size_" + str(self.brain.size)  + "_" + self.loadName + "_Time.csv",delimiter=",")
            print(self.brain.weights)

        self.setupSim()
        print(self.joint2NP.getHpr(self.joint1NP))
        

    def setupSim(self):
        self.cam.setPos(20,-20,20)
        self.cam.lookAt(0,0,5)
        self.dlnp.setPos(10,-10,10)
        self.dlnp.lookAt(0,0,0)
        self.paddleBotNP = self.loader.loadModel("models/paddleBot.bam")
        self.paddleBotNP.reparentTo(self.render)
        self.baseNP = self.paddleBotNP.find("Base")
        self.joint1NP = self.baseNP.find("Joint 1")
        self.joint2NP = self.joint1NP.find("Bone 1/Joint 2")
        self.paddleNP = self.joint2NP.find("Bone 2/Paddle")

        self.ballNP = self.loader.loadModel("models/ball.bam")
        self.ballNP.reparentTo(self.render)
        self.ballNP.setPos(0,10,5)
        self.ballNP.lookAt(self.render,0,0,5)
        
    def updateProcedures(self):
        self.simUpdate()
        # self.testingProcedure()
        self.diffEvolveProcedure()

    def resetBall(self):
        # self.ballNP.setPos(self.render,random.randint(-10,10),random.randint(-10,10),random.randint(0,10))
        self.ballNP.setPos(self.render,random.randint(-1,1),10,4)
        self.ballNP.setHpr(self.render,180,0,0)
        # self.ballNP.lookAt(self.render,random.randint(-2,2),random.randint(-2,2),random.randint(1,5))

    def resetSim(self):
        self.baseNP.setHpr(self.paddleBotNP,0,0,0)
        self.joint1NP.setHpr(self.baseNP,0,0,0)
        self.joint2NP.setHpr(self.joint1NP,90,0,0)
        self.currentScore = 0
        self.ballCounter = 0
        self.resetBall()


    def processBrain(self):
        inputArray = np.zeros(self.brain.size)
        inputArray[:3] = np.array(self.ballNP.getPos(self.render))
        inputArray[3] = self.baseNP.getH(self.paddleBotNP)
        inputArray[4] = self.joint1NP.getP(self.baseNP)
        inputArray[5] = self.joint2NP.getP(self.joint1NP)

        outputArray = self.brain.step(inputArray)
        self.baseNP.setHpr(self.baseNP,self.jointSpeed*outputArray[-3]*self.dt,0,0)
        self.joint1NP.setHpr(self.joint1NP,0,self.jointSpeed*outputArray[-2]*self.dt,0)
        self.joint2NP.setHpr(self.joint2NP,0,self.jointSpeed*outputArray[-1]*self.dt,0)

        if(self.joint1NP.getP(self.baseNP) > self.maxJointAngle):
            self.joint1NP.setHpr(self.baseNP,0,self.maxJointAngle,0)
        if(self.joint1NP.getP(self.baseNP) < -self.maxJointAngle):
            self.joint1NP.setHpr(self.baseNP,0,-self.maxJointAngle,0)

        if(self.joint2NP.getP(self.joint1NP) > self.maxJointAngle):
            self.joint2NP.setHpr(self.joint1NP,90,self.maxJointAngle,0)
        if(self.joint2NP.getP(self.joint1NP) < -self.maxJointAngle):
            self.joint2NP.setHpr(self.joint1NP,90,-self.maxJointAngle,0)

    def simUpdate(self):
        self.ballNP.setPos(self.ballNP,0,self.ballSpeed*self.dt,0)
        self.processBrain()


        if(self.ballNP.getPos(self.render).length()>18):
            self.resetBall()
            self.ballCounter+=1
            
        
        if(self.ballNP.getPos(self.paddleNP).length()<self.paddleSize):
            self.resetBall()
            self.ballCounter+=1
            self.currentScore+=1
        
    def getDifTestBrain(self):
        sample = list(range(len(self.pop)))
        random.shuffle(sample)
        donorList = []
        while(len(donorList)<3):
            testDonor = sample.pop()
            if(testDonor != self.individualIndex):
                donorList.append(testDonor)
        return CTRNN.generateDifferentialTestIndividual(self.pop[self.individualIndex][0],self.pop[donorList[0]][0],self.pop[donorList[1]][0],self.pop[donorList[2]][0],self.mut,self.crossRate)

    def diffEvolveProcedure(self):
        if self.ballCounter >= self.numDrops:
            
            if(self.gen == 0 or self.retest):
                self.pop[self.individualIndex][1] = self.currentScore
                
            else:
                # assume it has just tested the test vector
                if(self.currentScore > self.pop[self.individualIndex][1]):
                    self.pop[self.individualIndex][0] = self.brain
                    self.pop[self.individualIndex][1] = self.currentScore
            
            self.individualIndex += 1

            if(self.individualIndex == self.maxPop):

                self.individualIndex = 0
                self.pop.sort(key=lambda x : x[1], reverse=True)
                self.bestBrain = self.pop[0][0]
                self.bestBrainScore = self.pop[0][1]
                self.gen+=1

                if(self.gen%10 == 0):
                    print("Gen "+str(self.gen)+" best score is "+str(self.bestBrainScore))

                if(self.gen == self.maxGen):
                    print("Done. Best Score:",self.bestBrainScore)
                    if (self.saveBrain == True):
                        self.curateBrain()                         
                    sys.exit()
                
            if(self.gen == 0 or self.retest):
                self.brain = self.pop[self.individualIndex][0]
            else:
                self.brain = self.getDifTestBrain()
                        
            
      
            self.resetSim()
        
    def curateBrain(self):
        print("Saving Brain:",self.saveName)
        np.savetxt("results/brains/Size_" + str(self.brain.size)  + "_" +self.saveName + "_Weights.csv",self.bestBrain.weights,delimiter=",")
        np.savetxt("results/brains/Size_" + str(self.brain.size) + "_" + self.saveName + "_Bias.csv",self.bestBrain.bias,delimiter=",")
        np.savetxt("results/brains/Size_" + str(self.brain.size)  +  "_" + self.saveName + "_Time.csv",self.bestBrain.timescale,delimiter=",")
    
    def testingProcedure(self):
        if(self.ballCounter >= self.numDrops):
            print(self.currentScore)
            self.fitnessTrend[self.currentTrial] = self.currentScore/self.numDrops
            self.currentTrial += 1
            self.resetSim()


            if(self.currentTrial == self.numTrials):
                np.savetxt("results/"+self.saveName+".csv",self.fitnessTrend,delimiter=",")
                sys.exit()



        pass 

sim = paddleBot()
sim.run()
