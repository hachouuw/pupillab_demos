import zmq
import msgpack

topic = 'your_custom_topic'
payload = {'topic': topic}

# create and connect PUB socket to IPC
pub_socket = zmq.Socket(zmq.Context(), zmq.PUB)

ip = 'localhost'  # If you talk to a different machine use its IP.
port = 50020  # The port defaults to 50020. Set in Pupil Capture GUI.

ipc_pub_url = f'tcp://{ip}:{port}'


pub_socket.connect(ipc_pub_url)

# send payload using custom topic
pub_socket.send_string(topic, flags=zmq.SNDMORE)
pub_socket.send(msgpack.dumps(payload, use_bin_type=True))
print(f"{topic}: {payload}")