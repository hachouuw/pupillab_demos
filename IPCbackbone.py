import zmq
import time
import numpy as np
import msgpack
# specify the name of the surface you want to use
surface_name = "dell_window1"

ctx = zmq.Context()
# The REQ talks to Pupil remote and receives the session unique IPC SUB PORT
pupil_remote = ctx.socket(zmq.REQ)

ip = "127.0.0.1"   # If you talk to a different machine use its IP.
port = 50020  # The port defaults to 50020. Set in Pupil Capture GUI.

pupil_remote.connect(f'tcp://{ip}:{port}')

# Request 'SUB_PORT' for reading data
pupil_remote.send_string('SUB_PORT')
sub_port = pupil_remote.recv_string()

# open a sub port to listen to pupil
sub = ctx.socket(zmq.SUB)
sub.connect(f'tcp://{ip}:{sub_port}')
# sub.setsockopt_string(zmq.SUBSCRIBE, f"surfaces.{surface_name}")

# # Request 'PUB_PORT' for writing data
# pupil_remote.send_string('PUB_PORT')
# pub_port = pupil_remote.recv_string()

#...continued from above
# sub.subscribe("pupil.0.2d")  # receive gaze messages
# sub.subscribe('gaze.')  # receive all gaze messages
sub.subscribe('blink') 
# sub.subscribe('pupil.0.3d')  # receive all gaze messages
# sub.subscribe('pupil.1.3d')  # receive all gaze messages
i = 0
# while True:
t_end = time.time() + 10
while time.time() < t_end:
    topic, payload = sub.recv_multipart()
    message = msgpack.loads(payload)
    # print(f"{topic}: {message}")
    print('\n blink:',message[b'type'],message[b'timestamp'],message[b'timestamp']) 
    # print('\n gaze:',message[b'topic'],message[b'timestamp']) 
    # print('\n gaze:',message[b'topic'],message[b'timestamp'],message[b'norm_pos']) 
    
    # i+=1
    # print(i)

    # # diameters for each eye
    # diameter0 = message0[b'diameter']
    # diameter1 = message1[b'diameter']

    # # raw pupil positions of each eye
    # raw_pupil0 = message0[b'norm_pos']
    # raw_pupil1 = message1[b'norm_pos'] 
    # # raw_x, raw_y = 0.,0.

    # print(f"{topic}: {message}")
    # print('\n pupil 0 diameter:',message0[b'diameter'],'\n pupil 0 confidence:',message0[b'confidence'],'\n pupil 0 time:',message0[b'timestamp'])
    # print('\n pupil 1 diameter:',message1[b'diameter'],'\n pupil 1 confidence:',message1[b'confidence'],'\n pupil 1 time:',message1[b'timestamp'])

'''

try:
    topic, msg = sub.recv_multipart(flags=zmq.NOBLOCK)
    gaze_position = msgpack.loads(msg, raw=False)   
    
    gaze_on_screen = gaze_position['gaze_on_surfaces']

    # print(gaze_on_screen)

    # if len(gaze_on_screen) > 0:
    raw_x, raw_y = gaze_on_screen[-1]['norm_pos']

except:
    raw_x, raw_y = np.nan,np.nan

    # else:
    #     raw_x, raw_y = 0.,0.


else:
    raw_x, raw_y = 0.,0.

    # print('running',time.time())
    print('\n raw x: ', raw_x)
    # print('\n raw y: ', raw_y)
    # print('\n pupil 0 diameter:',message0[b'diameter'],'\n pupil 0 confidence:',message0[b'confidence'],'\n pupil 0 time:',message0[b'timestamp'])
    # print('\n pupil 1 diameter:',message1[b'diameter'],'\n pupil 1 confidence:',message1[b'confidence'],'\n pupil 1 time:',message1[b'timestamp'])
'''