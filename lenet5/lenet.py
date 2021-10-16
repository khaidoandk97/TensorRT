# Author: khaidoan
# This sample shows how to create and inference Lenet5 - TensorRT

import numpy as np
import pycuda.autoinit
import tensorrt as trt

# You can set the logger severity higher to suppress (or lower to display more message)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (1, 1, 32, 32)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32

def create_network():
    """
        Create 
    """