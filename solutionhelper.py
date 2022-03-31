from inspect import currentframe
import time
from turtle import width
import cv2
import numpy as np
from matplotlib import pyplot as plt 
from scipy import ndimage

class solution_helper: 
    # Class members
    start_frame = np.zeros((1024, 768))
    previous_frame = np.zeros((1024, 768))
    current_candidate_position = 0 
    previous_candidate_position = 0
    candidate_velocity = 0 


    def swap(self, tup):
        flipped = list(tup)
        flipped.reverse()
        return flipped


    def framediff(self, current_frame):  
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY) 
        diff = cv2.absdiff(current_frame, self.start_frame) 
        if not diff.any(): 
            return
        cv2.imwrite('Screenshots/diff.jpg', diff)
        _,thresh1 = cv2.threshold(diff,100,255,cv2.THRESH_BINARY)    
        cv2.imwrite('Screenshots/thresh.jpg', thresh1) 
        # Euclidean distance from background pixel to foreground pixel (Testing alternative location / bounding method) 
       
        # TODO maybe instead of distance can check out peak_local_max in sci py

        dist = ndimage.distance_transform_edt(thresh1)  
        cv2.imwrite('Screenshots/dist.jpg', dist*100)
        maxloc = np.where(dist == dist.max()) 
        print("number of distance max:",maxloc[0].shape)    
        for loc in zip(maxloc[0], maxloc[1]): 
            print("Duck center:",self.swap(loc)) 
            # Rows are columns in this bitch for some reason
            height_offset = 30
            width_offset = 50
            upperleft = (loc[1]-width_offset, loc[0]-height_offset) 
            bottomright = (loc[1]+width_offset, loc[0]+height_offset) 

            self.current_candidate_position = np.asarray(loc)

            cv2.rectangle(current_frame, upperleft, bottomright, 255, 2) 
            cv2.rectangle(thresh1, upperleft, bottomright, 255, 2) 
            cv2.imwrite('Screenshots/boundedframediff.jpg', current_frame)
            cv2.imwrite('Screenshots/boundedthreshold.jpg', thresh1)
            break  

        # Save prior frame, compute candidate velocity 
        self.previous_frame = current_frame 
        # Compute candidate velocity fom current and previous positions
        self.candidate_velocity = self.current_candidate_position - self.previous_candidate_position   
        print(self.candidate_velocity) 
        # Update previous position
        self.previous_candidate_position = self.current_candidate_position
        # Make a guess at future location of object
        predicted_location = self.current_candidate_position + self.candidate_velocity
        return predicted_location


    def GetLocation(self, move_type, env, current_frame): 
        # ***** Remove this later *****
        # time.sleep(.1) #artificial one second processing time 
        
        # Check if we have a level start frame
        if not self.start_frame.any(): 
            print('saving start frame')
            self.start_frame = np.swapaxes(cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY), 0, 1)
            cv2.imwrite('Screenshots/startframe.jpg', self.start_frame) 
            return [{'coordinate' : (0, 0), 'move_type' : 'relative'}]
        
        #Use relative coordinates to the current position of the "gun", defined as an integer below
        if move_type == "relative":
            coordinate = env.action_space.sample() 
        else:

            # Random coordinate for debugging (x,y)
            coordinate = (0, 0)

            # Current Frame shape: (1024, 768, 3) BGR   
    
            current_frame = np.swapaxes(current_frame,0,1) 

            coordinate = self.framediff(current_frame)  

            # Testing  
            # Flip tuple  
            coordinate = self.swap(coordinate) 
            print("Moving crosshair to:", coordinate)
            return [{'coordinate' : coordinate, 'move_type' : 'absolute'}] 
        
    
