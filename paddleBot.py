from lib.PyCTRNN import CTRNN
from lib.SimBase import SimBase
import sys
from panda3d.core import Vec3,DirectionalLight


class paddleBot(SimBase):

    def __init__(self, fStartDirect=True, windowType=None):
        super().__init__(fStartDirect, windowType)

    def setupSim(self):
        

        
sim = paddleBot()
sim.run()
