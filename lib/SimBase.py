from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
import simplepbr
import panda3d.bullet as bl
from panda3d.core import Vec3,DirectionalLight

class SimBase(ShowBase):

    def __init__(self, fStartDirect=True, windowType=None):
        super().__init__(fStartDirect, windowType)

        simplepbr.init()
        debugNode = bl.BulletDebugNode('Debug')
        debugNode.showWireframe(True)
        debugNode.showConstraints(True)
        debugNode.showBoundingBoxes(False)
        debugNode.showNormals(False)
        debugNP = self.render.attachNewNode(debugNode)
        debugNP.show()
        self.setFrameRateMeter(True)

        self.world = bl.BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))
        self.world.setDebugNode(debugNP.node())
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
        self.realTime = True
        self.taskMgr.add(self.update,"update")



    def update(self,task):

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
            self.updateProcedures()

        return task.cont

    def updateProcedures(self):
        pass


                       
               
        