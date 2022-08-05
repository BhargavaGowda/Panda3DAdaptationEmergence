from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
import panda3d.bullet as bl
from panda3d.core import Vec3,DirectionalLight,TransformState
import simplepbr
from puckDS import Puck


class Sim(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)
        simplepbr.init()
        debugNode = bl.BulletDebugNode('Debug')
        debugNode.showWireframe(True)
        debugNode.showConstraints(True)
        debugNode.showBoundingBoxes(False)
        debugNode.showNormals(False)
        debugNP = self.render.attachNewNode(debugNode)
        debugNP.show()
        self.setFrameRateMeter(True)
        self.simsteps = 10
        self.timestep = 0.016

        self.world = bl.BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))
        self.world.setDebugNode(debugNP.node())
        self.taskMgr.add(self.update,"update")

        self.cam.setPos(0,-15,15)
        self.cam.lookAt(0,0,0)

        dlight = DirectionalLight('sun')
        dlight.color = (2,2,2,1)
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setPos(10,0,10)
        dlnp.lookAt(0,0,0)
        self.render.setLight(dlnp)

        self.setUpStadium()
        self.setUpPucks(5)

       

    def setUpPucks(self,num):
        for i in range(num): 
            testPuck = Puck.makePuck(self.loader.loadModel("models/puckDS.bam"))
            self.render.attachNewNode(testPuck.bodyNP.node())
            self.world.attachRigidBody(testPuck.mainBodyNP.node())
            self.world.attachRigidBody(testPuck.leftWheelNP.node())
            self.world.attachRigidBody(testPuck.rightWheelNP.node())
            self.world.attachConstraint(testPuck.leftWheelConstraint)
            self.world.attachConstraint(testPuck.rightWheelConstraint)
            testPuck.leftWheelConstraint.enableAngularMotor(True,4,10)
            testPuck.rightWheelConstraint.enableAngularMotor(True,-4,10)
            testPuck.setPos(i*2,i*2,0)
        
        
    
        pass

    def setUpStadium(self):
        stadiumNP = self.loader.loadModel("models/stadium.bam")
        stadiumRBNP = self.render.attachNewNode(bl.BulletRigidBodyNode())
        stadiumNP.reparentTo(stadiumRBNP)
        stadiumRBNP.node().addShape(bl.BulletBoxShape(Vec3(22.5,22.5,0.1)))
        wallShape = bl.BulletBoxShape(Vec3(0.1,22.5,1))
        stadiumRBNP.node().addShape(wallShape,TransformState.makePos(Vec3(22.6,0,1)))
        stadiumRBNP.node().addShape(wallShape,TransformState.makePos(Vec3(-22.6,0,1)))
        stadiumRBNP.node().addShape(wallShape,TransformState.makePosHpr(Vec3(0,22.6,1),Vec3(90,0,0)))
        stadiumRBNP.node().addShape(wallShape,TransformState.makePosHpr(Vec3(0,-22.6,1),Vec3(90,0,0)))
        self.world.attachRigidBody(stadiumRBNP.node())
        stadiumRBNP.setPos(0,0,-3)

    def update(self,task):
        dt = globalClock.getDt()
        if(dt>1/130):
            self.simsteps-=1
        elif(dt<1/140):
            self.simsteps+=1
        for i in range(self.simsteps):
            self.world.doPhysics(self.timestep)
        return task.cont


    




app = Sim()
app.run()