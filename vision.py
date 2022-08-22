from lib.SimBase import SimBase
import panda3d.bullet as bl
from panda3d.core import Vec3,DirectionalLight
from lib.PyCTRNN import CTRNN

class VisionSim(SimBase):

    def __init__(self, fStartDirect=True, windowType=None):
        super().__init__(fStartDirect, windowType)

        


sim = VisionSim()
sim.run()