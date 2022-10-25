import socket
import time
import numpy as np
import struct
import pandas as pd

class QuattroOtlight():

    host = '127.0.0.1'
    port = 31000
    emg_channels = 384
    aux_channels = 16
    accessory_channels = 8
    sample_message_size = 0
    
    refresh_freq = 32 # change this to change the data streaming rate from Quattro
    sample_freq = 2048

    def __init__(self, chan_range = None) -> None:
        self.sample_message_size = self.emg_channels + self.aux_channels + self.accessory_channels
        self.sample_message_size_in_bytes = self.sample_message_size * 2 # two bytes per number
        self.batch_size = self.sample_freq // self.refresh_freq

        if chan_range: 
            self.output_channel_range = chan_range
        else:
            self.output_channel_range = range(self.emg_channels)

    def setup(self):
        server_addr = (self.host, self.port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.settimeout(None)  # setting this to None is blocking
            self.socket.connect(server_addr)
        except Exception as e:
            print("Quattrocento: error: {}".format(e))
            raise
        print(
            "Quattrocento: {} receiver subscribing to socket {}...".format(
                'quattro', server_addr))

        self.socket.send('startTX'.encode())
        data = self.socket.recv(8).decode() # read in 8 bytes to confirm connection
        if data != 'OTBioLab':
            raise Exception(f'cannot connect to the device at {server_addr}')
        
        print(f'connected to the device ')
        time.sleep(0.1)

    def read_emg(self):
        """
        read in a batch of emg  and decode into a tuple of 
        (emg_channels, aux_channels, sample_counter)
        """
        data = self.socket.recv(self.sample_message_size_in_bytes *self.batch_size,
                                socket.MSG_WAITALL )  # bytes to read
        dt = np.dtype('int16')
        dt = dt.newbyteorder('<')
        # data.shape here is (64, 408) - 408 = 384 (emg channels) + 16 (aux) + 8 (accessory)
        data = np.frombuffer(data, dtype=dt).reshape((-1, self.sample_message_size,))
        # data is (64, 408) - so datapoints x num channels
        

        return self._decode_emg(data)


    def _decode_emg(self, data:np.ndarray):
        """
        decompose data into actual emg, aux and accessory channels
        data is num_data_points by num_channels
        """
        emg_signals  = data[:, :self.emg_channels]
        aux_signals = data[:, self.emg_channels:(self.emg_channels+self.aux_channels)]
        sample_counter= data[:, -self.accessory_channels]

        #select output channels
        # emg_signals = emg_signals[:, self.output_channel_range]
        # select MULTI IN 1 channels
        # TODO: fix hard coding
        # emg_signals = num data x num channels
        emg_signals = emg_signals[:, 16*8:16*8+64]

        return (emg_signals, aux_signals, sample_counter)

    def tear_down(self):
        self.socket.send('stopTX'.encode())

        self.socket.close()