from inspect import currentframe
import time
from turtle import width
import cv2
import numpy as np
from matplotlib import pyplot as plt 
from scipy import ndimage

class solution_helper: 
    # Class members
    start_frame = np.zeros((256, 192)) 
    size = (256, 192)
    previous_frame = np.zeros((256, 192))
    currentblobs = [] 
    cbc = 0 
    pbc = 0 
    counter = 0
    previousblobs = np.array([]) 
    velocities = []  
    ffdiff = np.zeros((256, 192))
    predloc = 0  

    def framediff(self, current_frame):  
        
        # Compute diff between start and current frame to find foreground objects (was doing absdiff before)
        self.ffdiff = current_frame - self.previous_frame

        #Update previous frame 
        # cv2.imwrite('Screenshots/previousframe.jpg', self.previous_frame) 
        self.previous_frame = current_frame  

        # Apply a blur to the diff to eliminate crosshair movement and make blobs of ducks 
        diff = cv2.medianBlur(self.ffdiff, 3)
        # cv2.imwrite('Screenshots/diff.jpg', diff) 

        # Threshold diff to find blobs 
        threshold = 80
        _,thresh1 = cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY)   

        # Apply binary dilation to join noisy blobs
        # dilationstruct = ndimage.generate_binary_structure(2, 2)    # 3x3 dilation struct
        thresh1 = ndimage.binary_dilation(thresh1, iterations = 2).astype(thresh1.dtype) * 255
        # cv2.imwrite('Screenshots/thresh.jpg', thresh1)  

        # Connected components for image location, convenient for finding centroids
        ret = cv2.connectedComponentsWithStats(thresh1, 8, cv2.CV_32S)
        
        # Centroids are fourth index, update current blob count
        centroids = ret[3] 
        centroids = np.asarray(centroids[1:]).astype('int')
        self.currentblobs = centroids     
        self.cbc = self.currentblobs.shape[0] 

        # For debugging circle all blobs
        for location in self.currentblobs: 
            cv2.circle(thresh1, location, 10, 255, 1) 
        # cv2.imwrite('Screenshots/circledcentroids.jpg', thresh1)

        # Start case just go next
        if self.previousblobs.size == 0: 
            self.previousblobs = self.currentblobs   
            self.pbc = self.cbc
            # print("No previous blobs")
            return (0,0), 'relative'
       
        # If no new blobs
        if self.currentblobs.shape[0] == self.previousblobs.shape[0]:  
            return self.shootduck()
        
        # Otherwise for some change in blob count
        else: 
            # If current less blobs, one likely left screen
            if self.cbc < self.pbc:    
                # Restart check halfway through generated blobs
                # self.counter = self.cbc//2 
                self.counter = 0   
                # How many blobs left the screen
                bdiff = self.cbc - self.pbc  
                # Remove blobs that were closest to the bottom, subsequently at end of list
                self.previousblobs = self.previousblobs[:bdiff,:] 
                return self.shootduck()
            else:
                self.previousblobs = self.currentblobs
                self.pbc = self.cbc 
                self.counter = self.cbc//2
                return (0,0), 'relative'

    def shootduck(self): 
        # Calculate velocities
        self.velocities = self.currentblobs - self.previousblobs    
        # Update previous blobs and pbc
        self.previousblobs = self.currentblobs 
        self.pbc = self.previousblobs.shape[0] 
        # Predict blob locations based on velocity 
        predloc = self.currentblobs + self.velocities  
        # Catch case of no blobs
        if not self.velocities.any(): 
            return (0,0), 'relative'
        while True: 
            self.counter += 1 
            if self.counter < self.cbc: 
                if not self.velocities[self.counter, 0] > 50: 
                    return predloc[self.counter] * 4, 'absolute' 
                else: 
                    return (0,0), 'relative' 
            else: 
                self.counter = 0  
                if not self.velocities[self.counter, 0] > 50: 
                    return predloc[self.counter] * 4, 'absolute' 
                else: 
                    return (0,0), 'relative' 

    # # NOTE TESTING HTIS VERSION 
    # def shootduck(self): 
    #     # Calculate velocities
    #     self.velocities = self.currentblobs - self.previousblobs    
    #     # Update previous blobs and pbc
    #     self.previousblobs = self.currentblobs 
    #     self.pbc = self.previousblobs.shape[0] 
    #     # Predict blob locations based on velocity 
    #     predloc = self.currentblobs + self.velocities  
    #     # Catch case of no blobs
    #     if not self.velocities.any(): 
    #         return (0,0), 'relative'
    #     while True: 
    #         self.counter += 1 
    #         if self.counter < self.cbc: 
    #             if np.abs(self.velocities[self.counter, 0]) < 50: 
    #                 print(self.velocities[self.counter])
    #                 return predloc[self.counter] * 4, 'absolute' 
    #             else: 
    #                 # self.counter += 1 
    #                 return (0,0), 'relative' 
    #                 # continue
    #         else: 
    #             self.counter = 0  
    #             if np.abs(self.velocities[self.counter, 0]) < 50: 
    #                 print(self.velocities[self.counter])
    #                 return predloc[self.counter] * 4, 'absolute' 
    #             else: 
    #                 self.counter = 0 
    #                 return (0,0), 'relative' 



    def GetLocation(self, move_type, env, current_frame): 
        # ***** Remove this later *****
        # time.sleep(.1) #artificial one second processing time 
        
        # Check if we have a level start frame NOTE: Very first frame seems to be slightly discolored for some reason
        if not self.start_frame.any(): 
            print('Saving start frame')
            self.start_frame = np.swapaxes(cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY), 0, 1)   
            self.start_frame = cv2.resize(self.start_frame, self.size)  
            self.previous_frame = self.start_frame 
            cv2.imwrite('Screenshots/startframe.jpg', self.start_frame) 
            return [{'coordinate' : (0, 0), 'move_type' : 'relative'}]
        
        #Use relative coordinates to the current position of the "gun", defined as an integer below
        if move_type == "relative":
            coordinate = env.action_space.sample() 
        else:

            # Random coordinate for debugging (x,y)
            coordinate = (0, 0)

            # Current Frame shape: (256, 192, 3) BGR   
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)    
            current_frame = np.swapaxes(current_frame,0,1) 
            current_frame = cv2.resize(current_frame, self.size)  
            cv2.imwrite('Screenshots/currentframe.jpg', current_frame)

            coordinate, movetype = self.framediff(current_frame)  

            return [{'coordinate' : coordinate, 'move_type' : movetype}]  

        
    
