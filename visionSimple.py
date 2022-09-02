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

        
        

        # PARAMETERS
        self.brainSize = 7
        self.realTime = True
        self.numDrops = 20
        self.areaSize = 5
        self.paddleSize = 1
        self.maxGen = 500
        self.mut = 1
        self.crossRate = 0.5
        self.numTrials = 30
        self.saveBrain = True
        self.retrieveBrain = True
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
            # self.brain.mask = mask
            # self.brain.applyMask()
            print(self.brain.weights)

    def update(self, task):

        frameTime = globalClock.getDt()
        # self.dt = frameTime
        steps = 1
        if(not self.realTime):
            self.simPrintTimer += frameTime

            if self.simPrintTimer>5:
                self.simPrintTimer = 0
                print("Sim Steps per sec:", str(round(self.simSteps/frameTime)))
                print("Timefactor:" + str(round((self.simSteps/frameTime)*self.dt)))
                
            if(frameTime>1/20):
                self.simSteps-=10
            elif(frameTime<1/40):
                self.simSteps+=10

            steps = self.simSteps

        for i in range(steps):

            self.simUpdate()
            # self.simpleEvolutionProcedure()
            # self.popEvolveProcedure()
            self.testingProcedure()
            # self.diffEvolveProcedure()


                       
               
        return task.cont


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
            # if (self.ball.getPos(self.render)-self.paddle.getPos(self.render)).length()<2:
            #     # print("hit")
            #     self.currentScore+=1
            if(abs(self.ball.getX()-self.paddle.getX())<self.paddleSize and abs(self.ball.getY()-self.paddle.getY())<self.paddleSize):
                self.currentScore+=1

            self.ball.setPos(self.render,random.randint(-self.areaSize,self.areaSize),random.randint(-self.areaSize,self.areaSize),10)
            self.ballCounter+=1
        
    def resetSim(self):
        self.paddle.setPos(self.render,0,0,0)
        self.ball.setPos(self.render,random.randint(-self.areaSize,self.areaSize),random.randint(-self.areaSize,self.areaSize),10)  
        self.currentScore = 0
        self.ballCounter = 0

    def resetAgent(self):
        self.bestBrainScore = 0
        self.currentScore = 0
        self.prevMask = self.brain.mask
        self.brain = CTRNN(10)
        self.brain.mask = self.prevMask
        self.bestBrain = self.brain

    def curateBrain(self):
        print("Saving Brain:",self.saveName)
        np.savetxt("results/brains/Size_" + str(self.brain.size)  + "_" +self.saveName + "_Weights.csv",self.bestBrain.weights,delimiter=",")
        np.savetxt("results/brains/Size_" + str(self.brain.size) + "_" + self.saveName + "_Bias.csv",self.bestBrain.bias,delimiter=",")
        np.savetxt("results/brains/Size_" + str(self.brain.size)  +  "_" + self.saveName + "_Time.csv",self.bestBrain.timescale,delimiter=",")
        
    def popEvolveProcedure(self):

        if self.ballCounter >= self.numDrops:
            self.pop[self.individualIndex][1] = self.currentScore
            self.individualIndex+=1

            if(self.individualIndex == self.maxPop):
                self.individualIndex = 0
                self.pop.sort(key=lambda x : x[1], reverse=True)
                if(self.pop[0][1]>self.bestBrainScore):
                    self.bestBrain = self.pop[0][0]
                    self.bestBrainScore = self.pop[0][1]
                surviveNum = len(self.pop)//2
                i=0
                while(i<self.maxPop-surviveNum):
                    newBrain = CTRNN.recombine(self.pop[random.randint(0,surviveNum-1)][0],self.pop[random.randint(0,surviveNum-1)][0])
                    newBrain.mutate(mutationSize=self.mut)
                    self.pop[surviveNum+i][0] = newBrain
                    i+=1

                if(self.gen%10 == 0):
                    print("Gen "+str(self.gen)+" best score is "+str(self.bestBrainScore))
                
                self.gen+=1

                if(self.gen == self.maxGen):
                    print("Done. Best Score:",self.bestBrainScore)

                    if (self.saveBrain == True):
                        self.curateBrain()
                    
                    sys.exit()



            self.brain = self.pop[self.individualIndex][0]
            
            

            self.resetSim()

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




           

    # Conduct a number of evolution trials and record the results
    def simpleEvolutionProcedure(self):

        if self.ballCounter >= self.numDrops:

            # replace brain if better score
            if(self.currentScore > self.bestBrainScore):
                self.bestBrain = self.brain
                self.bestBrainScore = self.currentScore

            # make new brain and mutate it
            self.brain = CTRNN.recombine(self.bestBrain,self.bestBrain)
            self.brain.mutateSplit(mutationSize=self.mut, timeChangeSize=0.1)  
            self.gen += 1 
            # print(self.brain.weights)
            # print(self.brain.bias)

            # print scores every 100 gens
            if self.gen%100 == 0:
                print(self.gen,"current:",self.currentScore,"best:",self.bestBrainScore)
            
            self.resetSim()

            if(self.gen == self.maxGen):
                
                print("Best Score was", self.bestBrainScore)
                if (self.saveBrain == True):
                    self.curateBrain()

                sys.exit()


                # self.gen = 0
                # self.fitnessTrend[self.currentTrial] = self.bestBrainScore/self.numDrops     
                # print("Trial " +str(self.currentTrial)+" best score was "+str(self.bestBrainScore))
                # self.resetAgent()
                # self.currentTrial+=1
                # if(self.currentTrial == self.numTrials):
                #     np.savetxt("results/"+self.saveName+".csv",self.fitnessTrend,delimiter=",")
                #     sys.exit()      

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

        


sim = VisionSim()
sim.run()