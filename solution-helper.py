import time
import cv2
import numpy as np
from matplotlib import pyplot as plt 

past_frame = 0 

def store_past_frame(frame): 
    past_frame = frame