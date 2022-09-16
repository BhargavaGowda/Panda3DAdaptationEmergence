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
        self.realTime = True
        self.maxJointAngle = 60
        self.ballSpeed = 5
        self.brainSize = 7
        self.numDrops = 20
        self.paddleSize = 1
        self.maxGen = 500
        self.mut = 1
        self.crossRate = 0.5
        self.numTrials = 30
        self.saveBrain = True
        self.retrieveBrain = False
        self.maxPop = 10

        mask = np.zeros((self.brainSize,self.brainSize))
        mask[-2,:5] = 1
        mask[-1,:5] = 1
        # mask[5,:6] = 1
        # mask[6,:7] = 1 

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
        self.saveName = "DiffEvolHard"
        self.loadName = "DiffEvolHard"
        self.fitnessTrend = np.zeros(self.numTrials)
        if(self.retrieveBrain == True):
            print("Loading Brain:",self.loadName)
            self.brain.weights = np.loadtxt("results/brains/Size_" + str(self.brain.size)  + "_" + self.loadName + "_Weights.csv",delimiter=",")
            self.brain.bias = np.loadtxt("results/brains/Size_" + str(self.brain.size)  + "_" + self.loadName + "_Bias.csv",delimiter=",")
            self.brain.timescale = np.loadtxt("results/brains/Size_" + str(self.brain.size)  + "_" + self.loadName + "_Time.csv",delimiter=",")
            print(self.brain.weights)

        self.setupSim()
        

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

    def resetBall(self):
        # self.ballNP.setPos(self.render,random.randint(-10,10),random.randint(-10,10),random.randint(0,10))
        self.ballNP.setPos(self.render,random.randint(-5,5),10,random.randint(0,5))
        self.ballNP.lookAt(self.render,random.randint(-2,2),random.randint(-2,2),random.randint(1,5))

    def simUpdate(self):
        self.ballNP.setPos(self.ballNP,0,self.ballSpeed*self.dt,0)
        if(self.ballNP.getPos(self.render).length()>18):
            self.resetBall()
            
        
        if(self.ballNP.getPos(self.paddleNP).length()<0.5):
            self.resetBall()
            self.currentScore+=1
        


        
        pass


        
sim = paddleBot()
sim.run()
