import sys 
import os
sys.path.append(os.path.abspath("./"))
import SlimeMold
from SlimeMold import Environment
import numpy

environmentInputs = numpy.random.uniform(float(0), float(1), [10, 50])
targetOutputs = numpy.random.uniform(float(0), float(1), [9, 50])

environment = Environment(environmentInputs, targetOutputs)
environment.go()
