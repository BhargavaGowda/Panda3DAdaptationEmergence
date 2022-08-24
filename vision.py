import random
from lib.SimBase import SimBase
import panda3d.bullet as bl
from direct.showbase.ShowBaseGlobal import globalClock
from panda3d.core import Vec3,DirectionalLight,CardMaker,PerspectiveLens,ConfigVariableBool
import numpy as np
import cv2
from lib.PyCTRNN import CTRNN

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

        self.ball = self.loader.loadModel("models/ball.bam")
        self.paddle = self.loader.loadModel("models/paddle.bam")
        self.ball.reparentTo(self.render)
        self.paddle.reparentTo(self.render)
        self.ball.setPos(0,0,10)
        self.paddle.setPos(0,0,0)

        visionBuffer = self.win.makeTextureBuffer("eye",8,8,to_ram=True)
        self.visionTexture = visionBuffer.getTexture()
        eyeLens = PerspectiveLens()
        eyeCam = self.makeCamera(visionBuffer,lens=eyeLens)
        eyeCam.reparentTo(self.render)
        eyeCam.setPos(0,0,15)
        eyeCam.lookAt(0,0,0)
             
        cm = CardMaker("card")
        cm.setFrame(-0.25,0.25,-0.25,0.25)
        card = self.aspect2d.attachNewNode(cm.generate())
        card.setPos(-0.75,0,-0.75)
        card.setTexture(self.visionTexture)

        self.brain = CTRNN(76)
        self.currentScore = 0
        self.bestBrain = self.brain
        self.bestBrainScore = 0
        self.ballCounter = 0
        self.gen = 0
        self.setBrainMask()
        



        


    def setBrainMask(self):
        mask = np.zeros((self.brain.size,self.brain.size))
        mask[64,:64] = np.ones(64)
        mask[74,64] = 1
        mask[75,64] = 1
        self.brain.mask = mask
        print(mask[64,64])




    def update(self, task):
        dt = 0.032
        visTexData = self.visionTexture.getRamImage()
        eyeArray = np.frombuffer(visTexData,np.uint8)

        if(eyeArray.size != 0):
            eyeArray = np.reshape(eyeArray,(8,8,4))
            eyeArray = cv2.cvtColor(eyeArray[:,:,:3],cv2.COLOR_BGR2GRAY).astype(float).flatten()
            eyeArray /= 255
           
           
            inputArray = np.zeros(self.brain.size)
            inputArray[:eyeArray.size] = eyeArray
            outputArray = self.brain.step(inputArray)
            self.paddle.setPos(self.paddle,outputArray[-2]*dt,outputArray[-1]*dt,0)

            if(self.paddle.getX() > 3):
                self.paddle.setX(self.render,3)
            elif(self.paddle.getX()<-3):
                self.paddle.setX(self.render,-3)
            
            if(self.paddle.getY() > 3):
                self.paddle.setY(self.render,3)
            elif(self.paddle.getY()<-3):
                self.paddle.setY(self.render,-3)
        
        self.ball.setPos(self.ball,0,0,-5*dt)
        if(self.ball.getZ()<-1):
            if (self.ball.getPos(self.render)-self.paddle.getPos(self.render)).length()<2:
                # print("hit")
                self.currentScore+=1

            self.ball.setPos(self.render,random.randint(-3,3),random.randint(-3,3),10)
            self.ballCounter+=1
        
        if self.ballCounter >= 10:
            if(self.currentScore > self.bestBrainScore):
                self.bestBrain = self.brain
                self.bestBrainScore = self.currentScore
            
            self.brain = CTRNN.recombine(self.bestBrain,self.bestBrain)
            self.brain.mutate(0.1)
            self.paddle.setPos(self.render,0,0,0)
            self.gen += 1
            if(self.gen%10 == 0):
                print("best score:",self.bestBrainScore)
                print("__")
            self.currentScore = 0
            self.ballCounter = 0
            
               
        return task.cont

        


sim = VisionSim()
sim.run()