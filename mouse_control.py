"""
Stream Pupil gaze coordinate data using zmq to control a mouse with your eye.
Please note that marker tracking must be enabled, and in this example we have named the surface "screen."
You can name the surface what you like in Pupil capture and then write the name of the surface you'd like to use on line 17.
"""

## install dependencies
# pip3 install zmq msgpack pyuserinput

import zmq
from msgpack import loads
import subprocess as sp
from platform import system

import time
import pickle

import sys

try:
    from pymouse import PyMouse
except ImportError:
    msg = """
    Please install PyMouse from PyUserInput
    https://github.com/PyUserInput/PyUserInput

    pip install PyUserInput
    """
    
m = PyMouse()

# specify the name of the surface you want to use
surface_name = "dell_window1"

context = zmq.Context()

# open a req port to talk to pupil
addr = "127.0.0.1"  # remote ip or localhost
req_port = "50020"  # same as in the pupil remote gui


req = context.socket(zmq.REQ)
req.connect("tcp://{}:{}".format(addr, req_port))
# ask for the sub port
req.send_string("SUB_PORT")
sub_port = req.recv_string()
# print("recv:", sub_port)

# open a sub port to listen to pupil
sub = context.socket(zmq.SUB)
sub.connect("tcp://{}:{}".format(addr, sub_port))
sub.setsockopt_string(zmq.SUBSCRIBE, f"surfaces.{surface_name}")

smooth_x, smooth_y = 0.5, 0.5

# screen size
x_dim, y_dim = m.screen_size()
print("x_dim: {}, y_dim: {}".format(x_dim, y_dim))

import msgpack
raw_data = []

# while True:
t_end = time.time() + 20
while time.time() < t_end:
    topic, msg = sub.recv_multipart()
    gaze_position = msgpack.loads(msg, raw=False)

    # print(f"{topic}: {msg}")
    # print('gaze_position:',gaze_position)
    if gaze_position["name"] == surface_name:

        gaze_on_screen = gaze_position["gaze_on_surfaces"]
        if len(gaze_on_screen) > 0:

            # there may be multiple gaze positions per frame, so you could average them
            # raw_x = sum([i['norm_pos'][0] for i in gaze_on_screen])/len(gaze_on_screen)
            # raw_y = sum([i['norm_pos'][1] for i in gaze_on_screen])/len(gaze_on_screen)

            # or just use the most recent gaze position on the surface
            raw_x, raw_y = gaze_on_screen[-1]["norm_pos"]
            raw_x, raw_y = gaze_on_screen[-1]["norm_pos"]

            # print('raw:', raw_x, raw_y)
            raw_data.append([raw_x, raw_y])
            # pickle.dump([raw_x, raw_y], file)

            # smoothing out the gaze so the mouse has smoother movement
            smooth_gain = 1.25
            smooth_x += smooth_gain * (raw_x - smooth_x)
            smooth_y += smooth_gain * (raw_y - smooth_y)
            x = smooth_x
            y = smooth_y

            y = 1 - y  # inverting y so it shows up correctly on screen
            x *= int(x_dim)
            y *= int(y_dim)
            # PyMouse or MacOS bugfix - can not go to extreme corners because of hot corners?
            x = min(x_dim - 10, max(10, x))
            y = min(y_dim - 10, max(10, y))

            print("x,y = ", (x,y))

            # print "%s,%s\n" %(x,y)
            m.move(int(x), int(y))
#pickle file name
# file = open('mouse_xy.pkl', 'wb')
# pickle.dump(raw_data, file)        
# file.close()
# exit(1)    
    # else:
    #     print("no gaze on screen")
    #     exit(1)  
    #     # file.close()
    
