import pygame
import labgraph as lg
import asyncio
import time
import numpy as np
import zmq
import msgpack
import subprocess as sp
from platform import system
import time
import pickle
import sys

class stream(lg.Message):
    timestamp: float
    data: np.ndarray # consists of num_channls by number of samples
    aux_data: np.ndarray # 16 channel aux channel data
    sample_num:np.ndarray

class EyeGenConfig(lg.Config):
    sampling_frequency: float = 60.0
    left_button_clicked: bool = True  # for right or left dominat handed. left: True, right: False
    debug_flag: bool = True

class EyeCoordinateGenerator(lg.Node):
    OUTPUT = lg.Topic(stream) 
    config: EyeGenConfig

    '''
    For eye tracking, need Pupil Capture running in the background (after calibration in Pupil Capture)
    '''
    surface_name = "dell_window1" # specify the name of the surface you want to use
    context = zmq.Context()

    # open a req port to talk to pupil
    addr = "127.0.0.1"  # remote ip or localhost
    req_port = "50020"  # same as in the pupil remote gui
    req = context.socket(zmq.REQ)
    req.connect("tcp://{}:{}".format(addr, req_port)) #connect to eye tracker
    req.send_string("SUB_PORT") # ask for the sub port
    sub_port = req.recv_string()
    print("recv:", sub_port)

    # open a sub port to listen to pupil
    sub = context.socket(zmq.SUB)
    sub.connect("tcp://{}:{}".format(addr, sub_port))
    sub.setsockopt_string(zmq.SUBSCRIBE, f"surfaces.{surface_name}")
    x_dim,y_dim = (1900, 1120) #screen size
    smooth_x, smooth_y = 0.5, 0.5 #smoothing out the eye inputs
    
    @lg.publisher(OUTPUT) #output for manual stream node
    async def generate_mouse_coor(self) -> lg.AsyncPublisher:
        while True:
            '''
            Get raw x, y coordinates from eye tracker
            '''
            topic, msg = self.sub.recv_multipart()
            gaze_position = msgpack.loads(msg, raw=False)
            gaze_on_screen = gaze_position["gaze_on_surfaces"]
            raw_x, raw_y = gaze_on_screen[-1]["norm_pos"] #raw x, y coordinates from eye tracker
            smooth_gain = 1.25 # smoothing out the gaze so the mouse has smoother movement
            self.smooth_x += smooth_gain * (raw_x - self.smooth_x)
            self.smooth_y += smooth_gain * (raw_y - self.smooth_y)
            x = self.smooth_x
            y = self.smooth_y

            #generate x, y coorinate of cursor
            y = 1 - y  # inverting y so it shows up correctly on screen
            x *= int(self.x_dim)
            y *= int(self.y_dim)
            gaze_location = np.array((x, y), dtype=np.float)
            # print("eye: ", cursor_coor)

            '''
            Code for getting x, y, click from mouse
            '''            
            # (to work with hybrid decoder) add a third entry for original "click", stack together signals
            cursor_signal = np.append(gaze_location, [0.0])
            cursor_signal = cursor_signal[:, np.newaxis].T # make this a column vector
        
            yield self.OUTPUT, stream(
                timestamp=time.time(),
                data=cursor_signal,
                aux_data=np.zeros(1), # place holder to comply with the message template
                sample_num= np.zeros(1) # placeholder
            )
            
            await asyncio.sleep(1/self.config.sampling_frequency)