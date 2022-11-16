import zmq
import time
import numpy as np
# specify the name of the surface you want to use
surface_name = "dell_window1"

ctx = zmq.Context()
# The REQ talks to Pupil remote and receives the session unique IPC SUB PORT
pupil_remote = ctx.socket(zmq.REQ)

ip = 'localhost'  # If you talk to a different machine use its IP.
port = 50020  # The port defaults to 50020. Set in Pupil Capture GUI.

pupil_remote.connect(f'tcp://{ip}:{port}')

# Request 'SUB_PORT' for reading data
pupil_remote.send_string('SUB_PORT')
sub_port = pupil_remote.recv_string()

# open a sub port to listen to pupil
sub = ctx.socket(zmq.SUB)
sub.connect(f'tcp://{ip}:{sub_port}')
# sub.setsockopt_string(zmq.SUBSCRIBE, f"surfaces.{surface_name}")
sub.setsockopt_string(zmq.SUBSCRIBE, f"surfaces.{surface_name}")

# # Request 'PUB_PORT' for writing data
# pupil_remote.send_string('PUB_PORT')
# pub_port = pupil_remote.recv_string()

#...continued from above
# Assumes `sub_port` to be set to the current subscription port
subscriber0 = ctx.socket(zmq.SUB)
subscriber0.connect(f'tcp://{ip}:{sub_port}')
subscriber1 = ctx.socket(zmq.SUB)
subscriber1.connect(f'tcp://{ip}:{sub_port}')
# subscriber.subscribe('gaze.')  # receive all gaze messages
subscriber0.subscribe('pupil.0.3d')  # receive all gaze messages
subscriber1.subscribe('pupil.1.3d')  # receive all gaze messages
# subscriber.subscribe('pupil.')  # receive all pupil messages

# we need a serializer
import msgpack

# while True:
t_end = time.time() + 10
while time.time() < t_end:
    topic0, payload0 = subscriber0.recv_multipart()
    topic1, payload1 = subscriber1.recv_multipart()
    message0 = msgpack.loads(payload0)
    message1 = msgpack.loads(payload1)

    # diameters for each eye
    diameter0 = message0[b'diameter']
    diameter1 = message1[b'diameter']

    # raw pupil positions of each eye
    raw_pupil0 = message0[b'norm_pos']
    raw_pupil1 = message1[b'norm_pos'] 
    # raw_x, raw_y = 0.,0.

    # raw_pupil = np.concatenate(np.array((raw_pupil0), dtype = np.float), np.array((raw_pupil1), dtype = np.float)) # convert to np arrays 
    raw_pupil = np.array((raw_pupil0), dtype = np.float)
    raw_pupil = raw_pupil[:, np.newaxis].T

    # print(f"{topic}: {message}")
    # print('\n pupil 0 diameter:',message0[b'diameter'],'\n pupil 0 confidence:',message0[b'confidence'],'\n pupil 0 time:',message0[b'timestamp'])
    # print('\n pupil 1 diameter:',message1[b'diameter'],'\n pupil 1 confidence:',message1[b'confidence'],'\n pupil 1 time:',message1[b'timestamp'])
    # topic = sub.recv_string()
    # msg = sub.recv(flags=zmq.NOBLOCK)  # bytes
    # surfaces = msgpack.loads(msg, raw=False)

    # if surfaces["name"] == surface_name:

    # print('\n raw eye0: ', raw_pupil0)
    # print('\n raw eye1: ', raw_pupil1)
    # print('\n raw pupil data: ', raw_pupil)
    print(raw_pupil.shape)

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


    # else:
    #     raw_x, raw_y = 0.,0.

    # print('running',time.time())
    # print('\n raw x: ', raw_x)
    # print('\n raw y: ', raw_y)
    # print('\n pupil 0 diameter:',message0[b'diameter'],'\n pupil 0 confidence:',message0[b'confidence'],'\n pupil 0 time:',message0[b'timestamp'])
    # print('\n pupil 1 diameter:',message1[b'diameter'],'\n pupil 1 confidence:',message1[b'confidence'],'\n pupil 1 time:',message1[b'timestamp'])
