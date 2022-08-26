import random
from turtle import pensize
from lib.SimBase import SimBase
import panda3d.bullet as bl
from direct.showbase.ShowBaseGlobal import globalClock
from panda3d.core import Vec3,DirectionalLight,ConfigVariableBool
import numpy as np
import cv2
from lib.PyCTRNN import CTRNN
import sys

vsync = ConfigVariableBool("sync-video",True)

class VisionSim(SimBase):

    def __init__(self, fStartDirect=True, windowType=None):
        super().__init__(fStartDirect, windowType)


        dlight = DirectionalLight('sun')
        dlight.color = (4,4,4,1)
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setPos(10,0,10)
        dlnp.lookAt(0,0,0)
        self.render.setLight(dlnp)
        self.cam.setPos(0,0,30)
        self.cam.lookAt(0,0,0)
        self.simSteps = 10
        self.dt = 0.032
        self.simPrintTimer = 10

        self.ball = self.loader.loadModel("models/ball.bam")
        self.paddle = self.loader.loadModel("models/paddle.bam")
        self.ball.reparentTo(self.render)
        self.paddle.reparentTo(self.render)
        self.ball.setPos(0,0,10)
        self.paddle.setPos(0,0,0)


        self.brain = CTRNN(10)
        self.currentScore = 0
        self.bestBrain = self.brain
        self.bestBrainScore = 0
        self.ballCounter = 0
        self.numDrops = 20
        self.areaSize = 3
        self.maxGen = 500
        self.gen = 0
        self.mut = 1
        self.numTrials = 10
        self.currentTrial = 0
        self.fitnessTrend = np.zeros(self.numTrials)
        self.saveBrain = False
        self.saveName = "20Drop500Gen10Trials"
        # self.loadName = "Discrete20Fit"

        # self.brain.weights = np.loadtxt("results/brains/Size_10_" + self.loadName + "_Weights.csv",delimiter=",")
        # self.brain.bias = np.loadtxt("results/brains/Size_10_" + self.loadName + "_Bias.csv",delimiter=",")
        # self.brain.timescale = np.loadtxt("results/brains/Size_10_" + self.loadName + "_Time.csv",delimiter=",")

        mask = np.zeros((10,10))
        mask[-2,:5] = 1
        mask[-1,:5] = 1
        # mask[5,:6] = 1 
        self.brain.mask = mask
        self.brain.applyMask()
        print(self.brain.weights)

    def simUpdate(self):
        
        inputArray = np.zeros(self.brain.size)
        inputArray[:3] = np.array(self.ball.getPos(self.render))
        inputArray[3:5] = np.array([self.paddle.getX(self.render),self.paddle.getY(self.render)])
        outputArray = self.brain.step(inputArray)
        self.paddle.setPos(self.paddle,outputArray[-2]*self.dt,outputArray[-1]*self.dt,0)

        if(self.paddle.getX() > self.areaSize):
            self.paddle.setX(self.render,self.areaSize)
        elif(self.paddle.getX()<-self.areaSize):
            self.paddle.setX(self.render,-self.areaSize)
        
        if(self.paddle.getY() > self.areaSize):
            self.paddle.setY(self.render,self.areaSize)
        elif(self.paddle.getY()<-self.areaSize):
            self.paddle.setY(self.render,-self.areaSize)
        
        self.ball.setPos(self.ball,0,0,-5*self.dt)
        if(self.ball.getZ()<-1):
            # self.currentScore += 4-(self.ball.getPos(self.render)-self.paddle.getPos(self.render)).length()
            if (self.ball.getPos(self.render)-self.paddle.getPos(self.render)).length()<2:
                # print("hit")
                self.currentScore+=1

            self.ball.setPos(self.render,random.randint(-self.areaSize,self.areaSize),random.randint(-self.areaSize,self.areaSize),10)
            self.ballCounter+=1
        
        if self.ballCounter >= self.numDrops:
            self.evolveAgent()
            if self.gen%100 == 0:
                print(self.gen,"current:",self.currentScore,"best:",self.bestBrainScore)
            self.paddle.setPos(self.render,0,0,0)
            # self.fitnessTrend[self.gen] = self.bestBrainScore         
            self.currentScore = 0
            self.ballCounter = 0
            self.gen += 1

            # if (self.gen!=0 and self.gen%100==0):
            #     if(self.mut == 1):
            #         self.mut = 0.1
            #     else:
            #         self.mut = 1
            
            if(self.gen == self.maxGen):
                # np.savetxt("results/"+self.saveName+"Trend.csv",self.fitnessTrend,delimiter=",")
            
                self.fitnessTrend[self.currentTrial] = self.bestBrainScore/self.numDrops
                if (self.saveBrain == True):
                    np.savetxt("results/brains/Size_" + str(self.brain.size) + "_Trial_"+ str(self.currentTrial) + "_" + self.saveName + "_Weights.csv",self.bestBrain.weights,delimiter=",")
                    np.savetxt("results/brains/Size_" + str(self.brain.size) + "_Trial_"+ str(self.currentTrial)  + "_" + self.saveName + "_Bias.csv",self.bestBrain.bias,delimiter=",")
                    np.savetxt("results/brains/Size_" + str(self.brain.size) + "_Trial_"+ str(self.currentTrial) +  "_" + self.saveName + "_Time.csv",self.bestBrain.timescale,delimiter=",")
                print("Trial "+str(self.currentTrial)+" best score was "+str(self.bestBrainScore))

                self.currentTrial+=1
                if(self.currentTrial == self.numTrials):
                    np.savetxt("results/"+self.saveName+"Results.csv",self.fitnessTrend,delimiter=",")
                    sys.exit()
                
                self.gen = 0
                self.bestBrainScore = 0
                self.prevMask = self.brain.mask
                self.brain = CTRNN(10)
                self.brain.mask = self.prevMask
                self.bestBrain = self.brain

        
    def evolveAgent(self):
        if(self.currentScore > self.bestBrainScore):
            self.bestBrain = self.brain
            self.bestBrainScore = self.currentScore
            
        self.brain = CTRNN.recombine(self.bestBrain,self.bestBrain)
        self.brain.mutateSplit(mutationSize=self.mut, timeChangeSize=0.1)
        # print(self.brain.weights)
        # print(self.brain.bias)
        
        


    def update(self, task):
        frameTime = globalClock.getDt()
        if self.simPrintTimer>5:
            self.simPrintTimer = 0
            print("Sim Steps per sec:", str(round(self.simSteps/frameTime)))
            print("Timefactor:" + str(round((self.simSteps/frameTime)*self.dt)))
        if(frameTime>1/20):
            self.simSteps-=10
        elif(frameTime<1/40):
            self.simSteps+=10
        
        for i in range(self.simSteps):
            self.simUpdate()

        
        self.simPrintTimer += frameTime
        


        
            
                         
               
        return task.cont

        


sim = VisionSim()
sim.run()