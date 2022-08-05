from matplotlib.transforms import Transform
import panda3d.bullet as bl
from panda3d.core import Vec3,TransformState
from direct.showbase.Loader import Loader
from lib.PyCTRNN import CTRNN

#Puck bot with 2 distance sensors and 2 wheels
class Puck:

    def __init__(self,bodyNP,brain,maxSpeed,sensorRange):
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
        Vec3(0,0,0),Vec3(-0.54,0,-0.05),Vec3(0,0,0),Vec3(-1,0,0))
        self.rightWheelConstraint = bl.BulletHingeConstraint(self.rightWheelNP.node(),self.mainBodyNP.node(),
        Vec3(0,0,0),Vec3(0.54,0,-0.05),Vec3(0,0,0),Vec3(1,0,0))

    @staticmethod
    def makePuck(bodyNP,brainSize=4,maxSpeed=10,sensorRange=2):
        brain = CTRNN(brainSize)
        return Puck(bodyNP,brain,maxSpeed,sensorRange)

    def setPos(self,x,y,z):
        self.mainBodyNP.setPos(x,y,z)
        self.leftWheelNP.setPos(self.mainBodyNP,-0.54,0,-0.05)
        self.rightWheelNP.setPos(self.mainBodyNP,0.54,0,-0.05)
        
    
