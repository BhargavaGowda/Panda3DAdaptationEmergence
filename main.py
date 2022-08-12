
from random import randint, random, randrange
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
import panda3d.bullet as bl
from panda3d.core import Vec3,DirectionalLight,TransformState
import simplepbr
from lib.PyCTRNN import CTRNN
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
        self.timer1 = 0
        self.timerEvolve = 0
        self.puckNum = 14

        self.world = bl.BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))
        self.world.setDebugNode(debugNP.node())
        self.taskMgr.add(self.update,"update")

        self.cam.setPos(0,-60,45)
        self.cam.lookAt(0,0,0)

        dlight = DirectionalLight('sun')
        dlight.color = (2,2,2,1)
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setPos(10,0,10)
        dlnp.lookAt(0,0,0)
        self.render.setLight(dlnp)

        self.setUpStadium()
        self.puckList=[]
        self.setUpPucks(self.puckNum,self.puckList)

        self.goalBox = self.setUpGoal()
        self.world.attachRigidBody(self.goalBox.node())
        self.goalBox.setPos(15,15,1)
        

       

    def setUpPucks(self,num,puckList):
        for i in range(num): 
            testPuck = Puck.makePuck(self.world,self.loader.loadModel("models/puckDS.bam"))
            self.render.attachNewNode(testPuck.bodyNP.node())
            testPuck.setPos(-1*i-5,i-15,0)
            testbox1 = self.loader.loadModel("models/box.egg")
            testbox1.reparentTo(testPuck.ds1)
            testbox1.setColor(1,0,0,1)
            testbox2 = self.loader.loadModel("models/box.egg")
            testbox2.reparentTo(testPuck.ds2)
            testbox2.setColor(1,0,0,1)
            puckList.append(testPuck)

        
        
    
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
        stadiumRBNP.setPos(0,0,-1)

    def setUpGoal(self):
        goalBoxRBNP = self.render.attachNewNode(bl.BulletRigidBodyNode())
        goalBoxRBNP.node().addShape(bl.BulletBoxShape(Vec3(0.5,0.5,0.5)))
        goalBoxRBNP.node().setMass(2)
        goalBoxNP = self.loader.loadModel("models/box.egg")
        goalBoxNP.setPos(goalBoxRBNP,-0.5,-0.5,-0.5)
        goalBoxNP.setColor(200,0,0,1)
        goalBoxNP.reparentTo(goalBoxRBNP)
        return goalBoxRBNP

    def update(self,task):
        dt = globalClock.getDt()
        self.timer1 += dt
        self.timerEvolve += self.simsteps*self.timestep
        if(dt>1/20):
            self.simsteps-=1
        elif(dt<1/40):
            self.simsteps+=1
        for i in range(self.simsteps):
            for puck in self.puckList:
                sensorPos=puck.sensorBodyNP.getPos(self.render)
                ds1Pos = puck.ds1.getPos(self.render)
                ds2Pos = puck.ds2.getPos(self.render)
                ds1Hit = 1-self.world.rayTestClosest(sensorPos,ds1Pos).getHitFraction()
                ds2Hit = 1-self.world.rayTestClosest(sensorPos,ds2Pos).getHitFraction()
                puck.runPuck([ds1Hit,ds2Hit])

            self.world.doPhysics(self.timestep)
        if(self.timer1>1):
            self.timer1 = 0
            print(self.simsteps)

        if(self.timerEvolve>60):
            self.evolvePucks()
            self.timerEvolve=0

        return task.cont

    def evolvePucks(self):
        self.puckList.sort(key=lambda x: (self.goalBox.getPos(self.render)-x.mainBodyNP.getPos(self.render)).length())
        print((self.goalBox.getPos(self.render)-self.puckList[0].mainBodyNP.getPos(self.render)).length())
        surviveNum = len(self.puckList)-4
        # avgPos = self.puckList[surviveNum-1].mainBodyNP.getPos(self.render)+(self.puckList[0].mainBodyNP.getPos(self.render)-self.puckList[surviveNum-1].mainBodyNP.getPos(self.render))/2
        # print(avgPos)
        for i in range(4):
            newBrain = CTRNN.recombine(self.puckList[randrange(0,surviveNum)].brain,self.puckList[randrange(0,surviveNum)].brain)
            newBrain.mutate()
            testPuck = Puck.makePuck(self.world,self.loader.loadModel("models/puckDS.bam"))
            self.render.attachNewNode(testPuck.bodyNP.node())
            # pos = avgPos + Vec3(-1*i,i,0)
            # testPuck.setPosRel(self.render,pos.get_x(),pos.get_y(),2)
            testbox1 = self.loader.loadModel("models/box.egg")
            testbox1.reparentTo(testPuck.ds1)
            testbox1.setColor(1,0,0,1)
            testbox2 = self.loader.loadModel("models/box.egg")
            testbox2.reparentTo(testPuck.ds2)
            testbox2.setColor(1,0,0,1)
            testPuck.brain = newBrain
            changePoint = surviveNum+i
            self.puckList[changePoint].destroyPuck()
            self.puckList[changePoint]=testPuck

        for i in range(len(self.puckList)):
            self.puckList[i].setPos(-1*i-5,i-15,0)


            



    
app = Sim()
app.run()