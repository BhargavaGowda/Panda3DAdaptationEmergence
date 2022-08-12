
import panda3d.bullet as bl
from panda3d.core import Vec3,TransformState,PandaNode
from direct.showbase.Loader import Loader
from lib.PyCTRNN import CTRNN
import numpy as np

#Puck bot with 2 distance sensors and 2 wheels
class Puck:

    def __init__(self,world,bodyNP,brain,maxSpeed,sensorRange):
        self.world = world
        self.brain = brain
        self.maxSpeed = maxSpeed
        self.sensorRange = sensorRange
        self.bodyNP = bodyNP.find("mainBody")
        self.mainBodyNP = bodyNP.find("mainBody/mainBody")
        self.rightWheelNP = bodyNP.find("mainBody/rightWheel/rightWheel")
        self.leftWheelNP = bodyNP.find("mainBody/leftWheel/leftWheel")
        self.sensorBodyNP= bodyNP.find("mainBody/sensors")
        self.sensorBodyNP.reparentTo(self.mainBodyNP)
        self.mainBodyNP.reparentTo(self.bodyNP)
        self.leftWheelConstraint = bl.BulletHingeConstraint(self.leftWheelNP.node(),self.mainBodyNP.node(),
        Vec3(0,0,0),Vec3(-0.54,0,-0.01),Vec3(0,0,0),Vec3(-1,0,0))
        self.rightWheelConstraint = bl.BulletHingeConstraint(self.rightWheelNP.node(),self.mainBodyNP.node(),
        Vec3(0,0,0),Vec3(0.54,0,-0.01),Vec3(0,0,0),Vec3(1,0,0))
        self.world.attachRigidBody(self.mainBodyNP.node())
        self.world.attachRigidBody(self.leftWheelNP.node())
        self.world.attachRigidBody(self.rightWheelNP.node())
        self.world.attachConstraint(self.leftWheelConstraint)
        self.world.attachConstraint(self.rightWheelConstraint)
        # self.leftWheelNP.node().setAngularDamping(1)
        # self.rightWheelNP.node().setAngularDamping(1)

        self.ds1 = self.sensorBodyNP.attachNewNode(PandaNode("ds1"))
        self.ds1.setPos(self.sensorBodyNP,-5,10,0)
        self.ds2 = self.sensorBodyNP.attachNewNode(PandaNode("ds2"))
        self.ds2.setPos(self.sensorBodyNP,5,10,0)

        
        

    @staticmethod
    def makePuck(world,bodyNP,brainSize=4,maxSpeed=10,sensorRange=2):
        brain = CTRNN(brainSize)
        return Puck(world,bodyNP,brain,maxSpeed,sensorRange)

    def setPos(self,x,y,z):
        self.mainBodyNP.setPos(x,y,z)
        self.leftWheelNP.setPos(self.mainBodyNP,-0.54,0,-0.01)
        self.rightWheelNP.setPos(self.mainBodyNP,0.54,0,-0.01)

    def runPuck(self,inputs):
        # print(inputs)

        # if(inputs[0]-inputs[1]>0.01):
        #     self.rightWheelConstraint.enableAngularMotor(True,10,10)
        #     self.leftWheelConstraint.enableAngularMotor(True,10,10)

        # elif(inputs[0]-inputs[1]<-0.01):
        #     self.rightWheelConstraint.enableAngularMotor(True,-10,10)
        #     self.leftWheelConstraint.enableAngularMotor(True,-10,10)
   
        # else:
        #     self.rightWheelConstraint.enableAngularMotor(True,-5,10)
        #     self.leftWheelConstraint.enableAngularMotor(True,5,10)

        

        
        inputArray = np.zeros(self.brain.size)
        for i in range(len(inputs)):
            inputArray[i] = inputs[i]
        output = self.brain.step(inputArray)
        
        self.leftWheelConstraint.enableAngularMotor(True,10*output[-2],100)
        self.rightWheelConstraint.enableAngularMotor(True,10*output[-1],100)

    def destroyPuck(self):
        self.world.remove(self.leftWheelNP.node())
        self.world.remove(self.mainBodyNP.node())
        self.world.remove(self.rightWheelNP.node())
        self.world.remove(self.rightWheelConstraint)
        self.world.remove(self.leftWheelConstraint)
        self.leftWheelNP.removeNode()
        self.rightWheelNP.removeNode()
        self.sensorBodyNP.removeNode()
        self.mainBodyNP.removeNode()
        self.bodyNP.removeNode()
        self.ds1.removeNode()
        self.ds2.removeNode()



    
