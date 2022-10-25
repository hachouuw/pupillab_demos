import pygame
import labgraph as lg
import asyncio
import time
import numpy as np
from quattrocento_input import stream
import zmq
# from msgpack import loads
import msgpack
import subprocess as sp
from platform import system
from pymouse import PyMouse

import time
import pickle

import sys


'''OLD CODE, WORKS FOR HYBRID TASK DEMO EYE'''


class MouseMessage(lg.Message):
    cursor_coor: np.ndarray
    left_button_clicked: bool
    right_button_clicked: bool
    cursor_signal: np.ndarray
    time_stamp: float


class MouseGenConfig(lg.Config):
    sampling_frequency: float = 60.0
    left_button_clicked: bool = True  # for right or left dominat handed. left: True, right: False
    debug_flag: bool = True


class MouseCoordinateGenerator(lg.Node):

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
    x_dim,y_dim = (1280,720) #screen size
    smooth_x, smooth_y = 0.5, 0.5 #smoothing out the eye inputs

    OUTPUT = lg.Topic(stream) 
    generic_sensor_output = lg.Topic(MouseMessage) 
    config: MouseGenConfig

    pygame.init() #to have the mouse click control to work, but will not work in hybrid_teask_demos_eye because here is running a seperate pygame
    
    @lg.publisher(OUTPUT) #output for manual stream node
    @lg.publisher(generic_sensor_output)
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
            cursor_coor = np.array((x, y), dtype=np.float)
            # print("eye: ", cursor_coor)

            '''
            Code for getting x, y, click from mouse
            '''
            # cursor_coor = np.array(pygame.mouse.get_pos(), dtype=np.float) # code for getting mouse coordinates
            
            # get the button states
            pygame.event.get()
            button_states = pygame.mouse.get_pressed(num_buttons=3)
            button_states_array = np.array(list(map(lambda x: 1.0 if x else 0.0, button_states)))
            
            # stack together signals
            cursor_signal = np.append(cursor_coor, 
                                    button_states_array[0] if self.config.left_button_clicked else button_states_array[2])
            # make this a column vector
            cursor_signal = cursor_signal[:, np.newaxis].T
            
            if self.config.debug_flag and button_states[0]:
                print("MouseCoordinateGenerator::mouse button states: ", cursor_signal)
                print()
        
            yield self.OUTPUT, stream(
                timestamp=time.time(),
                data=cursor_signal,
                aux_data=np.zeros(1), # place holder to comply with the message template
                sample_num= np.zeros(1) # placeholder
            )
            
            yield self.generic_sensor_output, MouseMessage(
                cursor_coor=cursor_coor,
                left_button_clicked=button_states[0],
                right_button_clicked=button_states[2],
                cursor_signal=cursor_signal,
                time_stamp=time.time()
            )
            
            await asyncio.sleep(1/self.config.sampling_frequency)
            

class KeyMessage(lg.Message):
    time_stamp: float
    key_pressed: str


class KeyGenConfig(lg.Config):
    sample_rate: float = 30.0


class KeyGenerator(lg.Node):
    OUTPUT = lg.Topic(KeyMessage)
    config: KeyGenConfig

    def setup(self) -> None:
        # pygame.init()
        # screen = pygame.display.set_mode((640, 480))

        return super().setup()

    @lg.publisher(OUTPUT)
    async def generate_key(self) -> lg.AsyncPublisher:
        while True:
            yield self.OUTPUT, KeyMessage(
                time_stamp=time.time(),
                key_pressed=self.get_pygame_pressed()
            )
            await asyncio.sleep(1/self.config.sample_rate)

    def get_pygame_pressed(self):

        key_str = "not pressed"
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                key_str=event.unicode
                #print(key_str)
                return key_str

        
        return key_str
              

class KeyPrinter(lg.Node):
    INPUT = lg.Topic(KeyMessage)
    # A subscriber method that simply receives data and updates the node's state
    #TODO: convert it to print any message? 

    @lg.subscriber(INPUT)
    async def print_key_message(self, message: KeyMessage) -> lg.AsyncPublisher:
        if message.key_pressed != "not pressed": 
            print(f'received key pressed: {message.key_pressed} received at {message.time_stamp}')
        
        yield self.OUTPUT, message



