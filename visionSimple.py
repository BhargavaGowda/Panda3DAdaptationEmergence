import random
from lib.SimBase import SimBase
import panda3d.bullet as bl
from direct.showbase.ShowBaseGlobal import globalClock
from panda3d.core import Vec3,DirectionalLight,ConfigVariableBool
import numpy as np
import cv2
from lib.PyCTRNN import CTRNN
import sys

vsync = ConfigVariableBool("sync-video",False)

class VisionSim(SimBase):

    def __init__(self, fStartDirect=True, windowType=None):
        super().__init__(fStartDirect, windowType)


        dlight = DirectionalLight('sun')
        dlight.color = (4,4,4,1)
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setPos(10,0,10)
        dlnp.lookAt(0,0,0)
        self.render.setLight(dlnp)
        self.cam.setPos(0,0,15)
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

        # visionBuffer = self.win.makeTextureBuffer("eye",8,8,to_ram=True)
        # self.visionTexture = visionBuffer.getTexture()
        # eyeLens = PerspectiveLens()
        # eyeCam = self.makeCamera(visionBuffer,lens=eyeLens)
        # eyeCam.reparentTo(self.render)
        # eyeCam.setPos(0,0,15)
        # eyeCam.lookAt(0,0,0)
             
        # cm = CardMaker("card")
        # cm.setFrame(-0.25,0.25,-0.25,0.25)
        # card = self.aspect2d.attachNewNode(cm.generate())
        # card.setPos(-0.75,0,-0.75)
        # card.setTexture(self.visionTexture)

        self.brain = CTRNN(10)
        self.currentScore = 0
        self.bestBrain = self.brain
        self.bestBrainScore = 0
        self.ballCounter = 0
        self.maxGen = 1000
        self.gen = 0
        self.fitnessTrend = np.zeros(self.maxGen)
        self.mut = 1

        self.brain.weights = np.loadtxt("results/brains/Size_10_TConstTime_Weights.csv",delimiter=",")
        self.brain.bias = np.loadtxt("results/brains/Size_10_TConstTime_Bias.csv",delimiter=",")
        self.brain.timescale = np.loadtxt("results/brains/Size_10_TConstTime_Time.csv",delimiter=",")

        mask = np.zeros((10,10))
        mask[-2,:5] = 1
        mask[-1,:5] = 1


        
        
        self.brain.mask = mask
        self.brain.applyMask()
        print(self.brain.weights)

    def simUpdate(self):
        
        inputArray = np.zeros(self.brain.size)
        inputArray[:3] = np.array(self.ball.getPos(self.render))
        inputArray[3:5] = np.array([self.paddle.getX(self.render),self.paddle.getY(self.render)])
        outputArray = self.brain.step(inputArray)
        self.paddle.setPos(self.paddle,outputArray[-2]*self.dt,outputArray[-1]*self.dt,0)

        if(self.paddle.getX() > 3):
            self.paddle.setX(self.render,3)
        elif(self.paddle.getX()<-3):
            self.paddle.setX(self.render,-3)
        
        if(self.paddle.getY() > 3):
            self.paddle.setY(self.render,3)
        elif(self.paddle.getY()<-3):
            self.paddle.setY(self.render,-3)
        
        self.ball.setPos(self.ball,0,0,-5*self.dt)
        if(self.ball.getZ()<-1):
            self.currentScore += 4-(self.ball.getPos(self.render)-self.paddle.getPos(self.render)).length()
            # if (self.ball.getPos(self.render)-self.paddle.getPos(self.render)).length()<3:
            #     # print("hit")
            #     self.currentScore+=1

            self.ball.setPos(self.render,random.randint(-3,3),random.randint(-3,3),10)
            self.ballCounter+=1
        
        if self.ballCounter >= 10:   
            self.evolveAgent()
            if self.gen%50 == 0:
                print(self.gen,"current:",self.currentScore,"best:",self.bestBrainScore)
            self.paddle.setPos(self.render,0,0,0)
            self.fitnessTrend[self.gen] = self.bestBrainScore         
            self.currentScore = 0
            self.ballCounter = 0
            self.gen += 1

            # if (self.gen!=0 and self.gen%100==0):
            #     if(self.mut == 1):
            #         self.mut = 0.1
            #     else:
            #         self.mut = 1
            
            if(self.gen == self.maxGen):
                np.savetxt("results/TConstTimeTrend.csv",self.fitnessTrend,delimiter=",")
                np.savetxt("results/brains/Size_" + str(self.brain.size) + "_TConstTime_Weights.csv",self.bestBrain.weights,delimiter=",")
                np.savetxt("results/brains/Size_" + str(self.brain.size) + "_TConstTime_Bias.csv",self.bestBrain.bias,delimiter=",")
                np.savetxt("results/brains/Size_" + str(self.brain.size) + "_TConstTime_Time.csv",self.bestBrain.timescale,delimiter=",")
                print("best score was",str(self.bestBrainScore))
                sys.exit()
        
    def evolveAgent(self):
        if(self.currentScore > self.bestBrainScore):
            self.bestBrain = self.brain
            self.bestBrainScore = self.currentScore
            
        self.brain = CTRNN.recombine(self.bestBrain,self.bestBrain)
        self.brain.mutateSimple(self.mut)
        # print(self.brain.weights)
        # print(self.brain.bias)
        
        


    def update(self, task):
        frameTime = globalClock.getDt()
        # if(frameTime>1/20):
        #     self.simSteps-=10
        # elif(frameTime<1/40):
        #     self.simSteps+=10
        
        for i in range(1):
            self.simUpdate()

        
        self.simPrintTimer += 1
        if self.simPrintTimer%100 == 0:
            print("Sim Steps per frame:", str(self.simSteps))


        
            
                         
               
        return task.cont

        


sim = VisionSim()
sim.run()