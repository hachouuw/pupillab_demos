'''
    desc: test pupil labs core real-time data streaming with their API

    notes: 
    * need to open pupil capture first!
    * currently running with conda activate py368
    * read: https://docs.pupil-labs.com/developer/core/network-api/
'''
import zmq
import time

print('hello')

ctx = zmq.Context()
pupil_remote = zmq.Socket(ctx, zmq.REQ)
pupil_remote.connect('tcp://127.0.0.1:50020')

# start recording
pupil_remote.send_string('R')
# For every message that you send to Pupil Remote, you need to receive the response. If you do not call recv(), Pupil Capture might become unresponsive!
print('rcv 1:', pupil_remote.recv_string())

time.sleep(5)
pupil_remote.send_string('r')
print('rcv 2:', pupil_remote.recv_string())

