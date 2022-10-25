import labgraph as lg
import time
import numpy as np
import pandas as pd
# from quattrocento_input import stream
from pupillab_input import stream
from tracking_task import ReferencePosition
from utility import *
import asyncio

from collections import deque

from numpy_ringbuffer import RingBuffer
import hdfwriter
import datetime
from rate import Rate
from scipy.optimize import minimize,least_squares
import copy as copy
from hybrid_task_fsms import DiscreteClassifierFSM

ALMOST_ZERO_TOL = .0001 # almost zero tolerance
Default_number_of_sensors = 64
DEFAULT_NUM_STATES = 2
MAX_NUM_CHARACTERS_FOR_STATE_NAME = 32

SUPPORTED_FILTER_MODES = set(["pass_through", "filter_emg"])
SUPPORTED_CONT_DECODER_MODES = set([
    'pos_pass_through', # for mouse control task, pos mapped to cursor pos on the screen
    'velocity_based_weiner', # for velocity-based weiner filter
])
SUPPORTED_DISC_DECODER_MODES = set([
    'mouse_cursor_click',
    "constant_class",
    "analogue_gesture",
    "cheater"
])

MOUSE_CLICK_PIN = 2 # defines which mouse channel controls the click
MOUSE_CLICK_SIGNAL = 1 # if 1, it's a click 

# loop control stuff
ASYNCIO_SLEEP_TIME = 0.0
TARGET_LOCATION_RADIUS = 10

class decoder_output(lg.Message):
    timestamp: float
    output_position: np.ndarray
    cursor_color_code_int: int
    termination_command: bool


class EyeHybridDecoderConfig(lg.Config):
    
    subject_id: int = 0 # subject id for file naming
    one_d_task_flag: bool = False
    
    #  emg buffer settings
    N_SENSORS:int = 64
    emg_sampling_rate:int = 2048
    sensor_packet_rate: float = 32.0 # TODO, change this at the demo - # change this to change the data streaming rate from Quattro
    emg_buffer_time = 120 # seconds of data
    emg_ring_buffer_size:int = 32 * 100
    emg_gain: float = 1e-1 # default for delsys = 5e-3, TODO: recalibrate to quattro

    # basic decoder settings
    enable_cont_decoder: bool =  True
    cont_decoder_mode:str = 'velocity_based_weiner'
    batch_length: int = 204 # how many data samples does get_emg grab
    sampling_rate: float = 60.0 # 
    initial_cont_decoder: np.ndarray
    cycle_rate: float = 60.0
    total_num_cycles:int = int(60.0 * 60.0 * 2) #60 fps * seconds * minutes

    # discrete game settings
    default_discrete_class:int = 0
    debug_fsm_flag:bool = False
    
    # discrete decode settings
    enable_discrete_decoder: bool = True
    disc_decode_mode:str = "mouse_cursor_click"
    initial_disc_decoder:np.ndarray
    discrete_decode_window_seconds_after_property_change:float = 1.
    discrete_decode_feedback_time:float = 2.0
    discrete_batch_length: int = 820 # number of data points to grab from raw data
    debug_discrete_flag: bool = True
    DOWN_SAMPLED_FREQ:int = 60
    emg_decoding_fraction: float = 0.8 # fraction of amount of emg for decoding,taken into account of user's reaction time
    num_targets:int =  2 # number of classes to decoder
    
    # filter settings
    filter_mode:str = 'pass_through'

    # continuous clda settings
    decoder_adaptable: bool = False
    W_batch_length: int = 60 * 20 # determines update rate of decoder, 60 is sampling rate
    alpha: float = 0.5 # update rate of decoder transformation matrix, higher alpha = more old decoder  ### FIXME: CHANGE THIS FOR EXPERIMENTS - LEARNING RATE
    beta: float = 0.5 # update rate of state transition matrix
    lambdaE: float = 1e-1
    lambdaD: float = 1e-1
    lambdaF: float = 1e-1
    assist_level:float = 0.0

    # clda settings for discrete decoder
    discrete_decoder_adaptable: bool = False
    learning_batch: int = 8


    # hdf settings
    save_hdf_flag: bool = False
    save_final_decoder: bool = False
    data_dir:str = "./" # default to current dir
    decoder_id: int = 0 # decoder id for file naming
    hdf_save_path:str = '' #default to current directory
    hdf_save_filename:str = 'weiner_task_data_sl_20211221.h5' ### FIXME: CHANGE THIS FOR EXPERIMENTS - SAVING
    hdf_system_name:str = 'weiner'

    # misc things
    num_trials_for_counting_succusses:int = 8
    show_statistics_flag:bool = True
    window_size_in_pixels: tuple = (1920, 1080)
    window_size_in_cm:tuple = (36, 24)
    sync_to_emg = False
    verbose:bool = False
    debug_time: bool = False
    

class EyeHybridDecoderState(lg.State):
    decoded_position: np.ndarray = np.empty((2,1))# vector of decoded position, append current
    decoded_velocity: np.ndarray = np.empty((2,1))# decoded velocity
    # reference target properties
    reference_position: np.ndarray = np.empty((2,1))# target position
    synced_reference_position: np.ndarray = np.empty((2,1))# target position
    
    
    intended_velocity: np.ndarray = np.empty((2,1))# intended velocity
    velocity: np.ndarray =  np.empty((2,1)) # vector of velocity, append current
    sensor_buffer: np.ndarray = np.empty((Default_number_of_sensors,1))
    sensor_data_filter: np.ndarray = np.empty((Default_number_of_sensors, 1))
    W: np.ndarray =  np.empty((2, Default_number_of_sensors))

    H: np.ndarray = np.empty((2,2))
    W_history: np.ndarray = np.empty((2,Default_number_of_sensors , 1))
    bound_limit: int = 0
    
    # discrete decoder settings
    W_disc: np.array = np.empty((1, Default_number_of_sensors))
    W_history_disc: np.ndarray = np.empty((2,Default_number_of_sensors , 1))
    current_discrete_state:int = 0
    decoded_disc_states: np.ndarray = np.full((1,), False)
    synced_reference_class:np.ndarray  = np.empty((1,))
    current_reference_target_code_int:int = 0
    current_display_reference_code_int:int = 0
    last_reference_target_code_int:int = 0
    next_classify_time:float = 0.0

    # analogue gesture based state variables 
    disc_decoded_velocity: np.ndarray = np.empty((2,1))# decoded velocity

    # clda settings
    collected_raw_emg_windows: np.ndarray = np.empty((Default_number_of_sensors, 1))
    collected_emg_windows: np.ndarray = np.empty((Default_number_of_sensors, 1))
    disc_intended_velocity: np.ndarray = np.empty((2,1))# intended velocity
    
    
    
class EyeHybridDecoderNode(lg.Node):
    
    # input settings
    SENSOR_INPUT = lg.Topic(stream) # data stream from EMG, timestamp & sensor data
    REFERENCE = lg.Topic(ReferencePosition)
    # node settings
    config: EyeHybridDecoderConfig
    state: EyeHybridDecoderState
    # output messages
    CONTROL_OUTPUT = lg.Topic(decoder_output) # decoded position
    REFERENCE_OUTPUT = lg.Topic(ReferencePosition)


    def setup(self) -> None:
        
        self._setup_loop_time_control()
        self._initialize_sensor_circular_buffer()

        # continuous decoder setup
        self._initialize_state_arrays()
        self._setup_cont_decoder_properties()
        self._setup_clda_params()
        
        # discete decoder set up
        self._setup_discrete_task_properties()
        self._setup_discrete_decoder_properties()
        
        if self.config.save_hdf_flag: 
            self._setup_hdf_save()
        if self.config.sync_to_emg: 
            self._setup_nidaq_sync()
        self.graphics_termination = False

    @lg.publisher(CONTROL_OUTPUT)
    @lg.publisher(REFERENCE_OUTPUT)
    async def cycle(self) -> None:
        
        self._next_loop_time = time.time()
        
        for cycle_count in range(self.config.total_num_cycles):
            
            loop_start_time = time.time()
            
            # wait for next loop start time
            while loop_start_time <  self._next_loop_time:
                await asyncio.sleep(ASYNCIO_SLEEP_TIME)
                loop_start_time = time.time()
                
            # predict control output from filtered signal
            if self.config.enable_cont_decoder:
                # get the last self.config.batchlength of data
                self._raw_signal_for_continuous= self.get_emg_data(n_pts=self.config.batch_length)
                # filter raw data
                self._filtered_signal = self.filter_signal(self._raw_signal_for_continuous)
                self.emg_ring_buffer.append(self._filtered_signal)
                self.predict_continuous_trajectory(self._filtered_signal)
                self.bound_decoder_output()
                # calcluate the next intended velocity
                if self.config.decoder_adaptable and self.learner():
                    self.update() # checks if we need to update the decoder matrix
                if self.config.assist_level > 0: 
                    self.assist()
            
            if self.config.enable_discrete_decoder:
                self._prev_state = self._discrete_fsm.state
                self._discrete_fsm.fsm_tick() # discrete logic is embeded inside this
                   
                
                if self._check_state_from_emg_collection_to_feedback():
                    self._raw_signal_for_discrete = self.get_emg_data(n_pts=self.config.discrete_batch_length)
                    self._filtered_signal_for_classification = self.filter_signal_for_classification(self._raw_signal_for_discrete)
                    decoded_class  = self.classify_discrete_attribute(self._filtered_signal_for_classification)
                    
                    self.state.current_discrete_state = decoded_class
                    self._save_discrete_data_to_state_variables(self._filtered_signal_for_classification, 
                                                                decoded_class, 
                                                                self.state.current_reference_target_code_int)
                    
                    if self.config.discrete_decoder_adaptable and self.discrete_learner():
                        self.update_discrete_decoder()
                    
                    
                    if self.config.debug_discrete_flag:
                        print('decode time', time.time(), "raw signal shape", self._raw_signal_for_discrete.shape)
                        print()

                    if self.config.show_statistics_flag:
                        self.print_out_task_performance()


                if self._check_state_from_feedback_to_idle():
                    self._reset_current_discrete_state()


                        
            current_time = time.time()
            yield self.CONTROL_OUTPUT, decoder_output(timestamp = current_time,
                                                    output_position = self.state.decoded_position[:, -1],
                                                    cursor_color_code_int =  self.state.current_discrete_state,
                                                    termination_command = False)

            yield self.REFERENCE_OUTPUT, ReferencePosition(timestamp=current_time,
                                        reference_position=self.state.reference_position[:,-1],
                                        reference_discrete_code_int=self.state.current_reference_target_code_int
                                        )
                                        
            if self.config.save_hdf_flag: 
                self._save_to_hdf_table(timestamp = current_time,
                                        raw_emg = self._raw_signal_for_continuous,
                                        filtered_emg = self._filtered_signal,
                                        weiner_filter_w = self.state.W,
                                        weiner_filter_h = self.state.H, 
                                        decoded_position = self.state.decoded_position[:, -1],
                                        decoded_velocity = self.state.decoded_velocity[:, -1],
                                        intended_velocity = self.state.intended_velocity[:, -1], 
                                        reference  = self.state.reference_position[:, -1],
                                        discrete_state = self._discrete_fsm.state,
                                        disc_weiner_filter_w = self.state.W_disc,
                                        reference_discrete_code_int = self.state.current_reference_target_code_int,
                                        filtered_signal_for_discrete = self._filtered_signal_for_classification,
                                        decoded_discrete_state = self.state.current_discrete_state,
                                        discrete_succuss_rate = self.succuss_rate)
            
            # lines for timing control
            loop_finish_time = time.time()
            loop_time = loop_finish_time - loop_start_time
            left_over_time = self._loop_duration - loop_time
            self._next_loop_time = time.time() +  max(left_over_time,0) 
            if self.config.debug_time:
                print(cycle_count, ": loop_finish_time", loop_finish_time)
                
        self.graphics_termination = True
        yield self.CONTROL_OUTPUT, decoder_output(timestamp = current_time,
                                        output_position = self.state.decoded_position[:, -1],
                                        cursor_color_code_int = bool(self.state.decoded_disc_states[-1]),
                                        termination_command = self.graphics_termination)
        print('exiting from the decoder node')
        raise lg.NormalTermination
    
    def cleanup(self) -> None:
        
        #save config attrs
        if self.config.save_hdf_flag:
            config_attrs = self.config.asdict()
            for key, value in config_attrs.items():
                self._hdf_writer.sendAttr(self.config.hdf_system_name, key, value)
                
            self._hdf_writer.close()
            
        if self.config.save_final_decoder:
            hdf_name = self.config.hdf_save_filename
            # get the file name without the file suffix
            file_name, h5_suffix = hdf_name.split(".")
            init_decoder_file_name = file_name + "_calibrated_decoder_initialization"
            
            with open(f"{self.config.hdf_save_path}\{init_decoder_file_name}",  "wb") as f:
                np.save(f, self.state.W)

        if self.config.sync_to_emg:
            self._nidaq_sync.clean_up()


    def _initialize_state_arrays(self):
    
        # random initialization of W
        self.state.W = self.config.initial_cont_decoder
        self.state.H = [[0, 0], [0, 0]] # initialize state transition matrix

        self.state.W_history = np.empty((2,self.config.N_SENSORS, 1))
        self.state.W_history[:, :, 0] = self.state.W

        # random initializaton of discrete decoder
        self.state.W_disc = self.config.initial_disc_decoder
        self.state.decoded_disc_states = np.empty((1,), dtype = int)
        self.state.synced_reference_class = np.empty((1,), dtype = int)
        # TODO: setup disc decoder saving

        self.state.decoded_position = np.zeros((2 , 1))
        self.state.reference_position = np.zeros((2 , 1))
        self.state.synced_reference_position = np.zeros((2 , 1))
        self.state.intended_velocity = np.zeros((2, 1))
        self.state.decoded_velocity = np.zeros((2, 1))
        self.idx_batch = 0
        self.state.bound_limit = 0

        self.state.sensor_buffer = np.empty((self.config.N_SENSORS, 1))
        self.state.sensor_data_filter = np.empty((self.config.N_SENSORS, 1))
        
    def _initialize_sensor_circular_buffer(self):
    
        #set up a circular buffer
        self.emg_buffer_size = int(self.config.emg_buffer_time * \
                               self.config.sensor_packet_rate * \
                               self.config.N_SENSORS)

        self.emg_circular_buffer = np.empty((self.config.N_SENSORS, 
                                        self.emg_buffer_size))
        self.emg_circular_buffer_pointer = 0
        
        # See: https://pypi.org/project/numpy_ringbuffer/#description for ring buffer initialization information
        self.emg_ring_buffer = RingBuffer(capacity=self.config.emg_ring_buffer_size, dtype=(float, self.config.N_SENSORS))

    def _setup_loop_time_control(self):
        
        # set up loop control
        self._loop_duration = 1 / self.config.cycle_rate
        self.last_time = time.time()
        if self.config.verbose: 
            print(f'each streaming time takes {self._loop_duration}s ')
            
    def _setup_clda_params(self):

        pass
        
    def _setup_cont_decoder_properties(self):
        if self.config.cont_decoder_mode not in SUPPORTED_CONT_DECODER_MODES:
            raise Exception("Not supported decoder mode", self.config.cont_decoder_mode)

        self._raw_signal_for_continuous = np.zeros((self.config.N_SENSORS, self.config.batch_length))
        self._filtered_signal = np.zeros((self.config.N_SENSORS))
        
    def _setup_discrete_task_properties(self):
        
        self.state.current_reference_target_code_int = 0
        self.state.last_reference_target_code_int = 0
        
        current_time = time.time()
        self.state.next_classify_time=current_time
        
        # set up the fsm
        self._discrete_fsm = DiscreteClassifierFSM(self)
        self._prev_state = self._discrete_fsm.state
        self._current_state = self._discrete_fsm.state

        # set up task statistics
        self.success_count = 0 # calculated once per self.config.num_trials_for_counting_succusses
        self.succuss_rate = 0
        
        # check decoder modes
        if self.config.disc_decode_mode not in SUPPORTED_DISC_DECODER_MODES:
            raise Exception(f"My applogies, {self.config.disc_decode_mode} is not supported, yet!")
        

    def _setup_discrete_decoder_properties(self):
        # set up discrete data saving
        self._raw_signal_for_discrete = np.zeros((self.config.N_SENSORS, self.config.discrete_batch_length))
        self.state.collected_raw_emg_windows = np.empty((self.config.N_SENSORS,
                                                     self.config.discrete_batch_length)) # initialize storage of task emg windows
        self._filtered_signal_for_classification = np.zeros((self.config.N_SENSORS, self.config.discrete_batch_length))

        
        if self.config.disc_decode_mode == "analogue_gesture":

            thetas = 2*np.pi/self.config.num_targets*np.arange(0,self.config.num_targets) + np.pi / 2# convert number of targets to angles
            
            # neutral position
            zeros = np.zeros((1,2))
            self.target_positions = TARGET_LOCATION_RADIUS*np.asarray([np.cos(thetas),np.sin(thetas)]).T
            self.target_positions = np.vstack((zeros, self.target_positions))

            print("set up discrete decoder properties: (imaginery target locations)", self.target_positions)

            self.win_downsample = int( np.floor(self.config.discrete_batch_length/self.config.emg_sampling_rate*self.config.DOWN_SAMPLED_FREQ) \
                                    * (self.config.emg_decoding_fraction))
                                    
            print("_setup_discrete_decoder_properties: self.win_downsample:", self.win_downsample)
            print()
            assert self.win_downsample > 0
            
            self.num_win_downsamples = self.win_downsample - 1
            
            # this is the filtered data
            self._raw_signal_for_discrete = np.zeros((self.config.N_SENSORS, self.num_win_downsamples))
            self.state.collected_emg_windows = np.empty((self.config.N_SENSORS, 1)) # initialize storage of task emg windows
            self._filtered_signal_for_classification = np.zeros((self.config.N_SENSORS, self.num_win_downsamples))
            self.state.disc_decoded_velocity= np.empty((2,1))
            
            # learning batch
            self.disc_idx_batch = 0
        else:
            thetas = 2*np.pi/self.config.num_targets*np.arange(0,self.config.num_targets) + np.pi / 2# convert number of targets to angles
            
            # neutral position
            zeros = np.zeros((1,2))
            self.target_positions = TARGET_LOCATION_RADIUS*np.asarray([np.cos(thetas),np.sin(thetas)]).T
            self.target_positions = np.vstack((zeros, self.target_positions))

            print("set up discrete decoder properties: (imaginery target locations)", self.target_positions)

            # self.win_downsample = int( np.floor(self.config.discrete_batch_length/self.config.emg_sampling_rate*self.config.DOWN_SAMPLED_FREQ) \
            #                         * (self.config.emg_decoding_fraction))
            self.win_downsample = int( np.floor(self.config.discrete_batch_length/self.config.DOWN_SAMPLED_FREQ) )
                                    
            print("_setup_discrete_decoder_properties: self.win_downsample:", self.win_downsample)
            print()
            assert self.win_downsample > 0
            
            self.num_win_downsamples = self.win_downsample - 1
            
            # this is the filtered data
            self._raw_signal_for_discrete = np.zeros((self.config.N_SENSORS, self.num_win_downsamples))
            self.state.collected_emg_windows = np.empty((self.config.N_SENSORS, 1)) # initialize storage of task emg windows
            self._filtered_signal_for_classification = np.zeros((self.config.N_SENSORS, self.num_win_downsamples))
            self.state.disc_decoded_velocity= np.empty((2,1))
            
            # learning batch
            self.disc_idx_batch = 0

    
    def _check_state_from_emg_collection_to_feedback(self):
        if self._prev_state == 'emg_collection' and self._discrete_fsm.state == 'feedback':
            self._prev_state = 'feedback'
            return True
        return False
    
    def _check_state_from_feedback_to_idle(self):
        if self._prev_state == 'feedback' and self._discrete_fsm.state == 'idle':
            self._prev_state = 'idle'
            return True
        return False


    def _check_in_the_same_state(self):
        return self._prev_state == self._discrete_fsm.state
    
    def _turn_off_discrete_change_flag(self):
        self.state.current_discrete_state = False

    def _reset_current_discrete_state(self):
        self.state.current_discrete_state = self.config.default_discrete_class
        #self.state.current_reference_target_code_int = self.config.default_discrete_class
        
    @lg.subscriber(SENSOR_INPUT)
    async def store_sensor_properties(self, message:stream) -> None: # collect learner batches (sensor input & intended output)
        # message.data = (num data x num channels)
        emg_data = (message.data).T

        #calculate pointer
        emg_sensor_count, emg_length = emg_data.shape
        idx, n_pts, max_len = self.emg_circular_buffer_pointer, emg_length, self.emg_buffer_size

        # TODO: possibly implement numpy_ringbuffer
        if idx + emg_length> max_len:
            self.emg_circular_buffer[:, idx:] = emg_data[:max_len-idx]
            self.emg_circular_buffer[:, :n_pts-(max_len-idx)] = emg_data[max_len-idx:]
            idx = n_pts-(max_len-idx)

        else:
            self.emg_circular_buffer[:, idx:idx+n_pts] = emg_data
            idx = idx+n_pts
            if idx == self.emg_buffer_size: 
                idx = 0

        self.emg_circular_buffer_pointer = idx
        
        await asyncio.sleep(1/32.0)

    def get_emg_data(self, n_pts = None):
        """
        pulling data from the emg circular buffer
        """
        if not n_pts: 
            n_pts = self.config.batch_length

        data = np.zeros((self.config.N_SENSORS, n_pts), dtype=np.float)

        idx = self.emg_circular_buffer_pointer
        if idx >= n_pts:  # no wrap-around required
            data = self.emg_circular_buffer[:, idx-n_pts:idx]
        else:
            data[:, :n_pts-idx] = self.emg_circular_buffer[:, -(n_pts-idx):]
            data[:, n_pts-idx:] = self.emg_circular_buffer[:, :idx]

        return data


    @lg.subscriber(REFERENCE)
    async def store_reference_properties(self, message:ReferencePosition):
        incoming_postion = message.reference_position.reshape(-1, 1)
        self.state.reference_position = np.append(self.state.reference_position, incoming_postion, axis = 1)
        
        # change reference target color

        self.state.current_reference_target_code_int = message.reference_discrete_code_int
        
        await asyncio.sleep(1/60.0)

    def filter_signal(self, raw_signal):
        
        if self.config.filter_mode in SUPPORTED_FILTER_MODES:
            if self.config.filter_mode == 'pass_through':
                return raw_signal.T
            elif self.config.filter_mode == 'filter_emg':
                filtered_signal = filt_fir(raw_signal)
                return filtered_signal
        else:
            raise Exception("Unsupported filter modes:", self.config.filter_mode)

    def predict_continuous_trajectory(self, sensor_data):
        
        # assign the right decoder to the given mode
        if self.config.cont_decoder_mode == "velocity_based_weiner":
            (position, velocity) = self.predict_use_weiner_velocity(sensor_data)
        elif self.config.cont_decoder_mode == "pos_pass_through":
            (position, velocity) = self.predict_by_direct_position_mapping(sensor_data)
            
        
        # save to the state variables
        self.state.decoded_position = np.append(self.state.decoded_position, position, axis = 1) # appends p+
        self.state.decoded_velocity = np.append(self.state.decoded_velocity, velocity, axis = 1) # appends v+
        self.state.synced_reference_position = np.append(self.state.synced_reference_position, self.state.reference_position[:, -1][:, np.newaxis], axis = 1)

    
    def filter_signal_for_classification(self, raw_signal):
        # just filter the signal and return filtered signal
        
    
        if raw_signal.shape != (self.config.N_SENSORS, self.config.discrete_batch_length):
            raise Exception(f"expected the raw signal to be {(self.config.N_SENSORS, self.config.discrete_batch_length)}")


        # MOMONA FILTERING AND PREDICTION
        # filter EMG signals from 2048 Hz to 60 Hz AND predict on each of filtered signals
        # this should be 60 * number of seconds



        window = np.floor(self.config.emg_sampling_rate/self.config.DOWN_SAMPLED_FREQ)
        
        filtered_signals = np.zeros((self.config.N_SENSORS, self.win_downsample - 1))

        # TODO: why is this self.win_downsample - 1
        for ix in range( self.win_downsample - 1):
            # window the signal
            startix = ix + int( np.floor(raw_signal.shape[1]/self.config.emg_sampling_rate*self.config.DOWN_SAMPLED_FREQ) \
                                        * (1 - self.config.emg_decoding_fraction)) # set offset for reaction time
            
            window_signal = raw_signal[:,int(startix*window):int((startix+1)*window)]

            # filter the windowed signal
            filtered_signal = self.filt_delinearize(window_signal)  # output is 64 x 1
            
            filtered_signals[:, ix] = filtered_signal



        return filtered_signals

        
    
    
    def classify_discrete_attribute(self, filtered_sensor_data):
        
        if self.config.disc_decode_mode == "mouse_cursor_click":
            decoded_discrete_class = self.classify_mouse_click(filtered_sensor_data)
        elif self.config.disc_decode_mode == "constant_class":
            decoded_discrete_class = 1
        elif self.config.disc_decode_mode == "cheater":
            decoded_discrete_class = self.state.current_reference_target_code_int
        elif self.config.disc_decode_mode == "analogue_gesture":
            decoded_discrete_class = self.classify_by_analogue_gesture(filtered_sensor_data)
        else:
            raise NotImplementedError(f'Hold on...{self.config.disc_decode_mode} is not implemented')


        if self.config.debug_discrete_flag and decoded_discrete_class:
            print("detected change in the decoder:")

        assert type(decoded_discrete_class) == int
            
        return decoded_discrete_class
    
    def _save_discrete_data_to_state_variables(self, temp_filtered_data_discrete:np.ndarray, classified_state, target_class):
        
        if temp_filtered_data_discrete.shape != (self.config.N_SENSORS, 
                                           self.num_win_downsamples):
            raise Exception(f"Expect discrete data window to be number of sensors by number of samples")

        # append to state variables
        self.state.collected_emg_windows = np.append(self.state.collected_emg_windows, temp_filtered_data_discrete, axis = 1)
        self.state.decoded_disc_states = np.append(self.state.decoded_disc_states, classified_state) # appends the color change flag
        self.state.synced_reference_class = np.append(self.state.synced_reference_class, target_class)



    def predict_use_weiner_velocity(self, sensor_data):
        sensor_data_filter = sensor_data
        sensor_data_filter = np.squeeze(sensor_data_filter)[:, np.newaxis]

        # velocity updates
        velocity_sensor_update = np.squeeze(self.state.W @ sensor_data_filter)
        velocity_prior_state_update = np.squeeze(self.state.H @ self.state.decoded_velocity[:, -1])
        velocity = np.squeeze(velocity_sensor_update + velocity_prior_state_update) # make sure the shapes match
        
        # pos update
        position = np.squeeze(self.state.decoded_position[:, -1]) \
            + (np.squeeze(self.state.decoded_velocity[:,-1]) / self.config.sampling_rate)  # p+ = p + (v+)*dt
        
        position = position[:,np.newaxis]
        velocity = velocity[:,np.newaxis]

        if self.config.one_d_task_flag:
            position[1] = 0
            velocity[1] = 0        

        return (position, velocity)
    
    def predict_by_direct_position_mapping(self, sensor_data):
        
        velocity = np.zeros((DEFAULT_NUM_STATES, 1))
        sensor_data_transpose = sensor_data.T
        position = self.state.W @ sensor_data_transpose
        
        # this needs to be scaled
        position = self.calculate_mouse_coordinate_2D(position)
        return (position, velocity)
        
    def classify_mouse_click(self, sensor_data):
        
        if sensor_data.shape != (self.config.N_SENSORS, self.config.discrete_batch_length):
            raise ValueError("Imcompatible sensor data shape:",  sensor_data.shape)
        
        max_over_batch = np.max(sensor_data, axis = 1)
        
        discrete_change_flag = abs(max_over_batch[MOUSE_CLICK_PIN] -  MOUSE_CLICK_SIGNAL ) <= ALMOST_ZERO_TOL 
        
        return discrete_change_flag
    
    def classify_by_analogue_gesture(self, filtered_signals):

        if filtered_signals.shape != (self.config.N_SENSORS, self.num_win_downsamples):
            raise Exception(f"expected the raw signal to be {(self.config.N_SENSORS, self.config.discrete_batch_length)}")


        # MOMONA FILTERING AND PREDICTION
        # filter EMG signals from 2048 Hz to 60 Hz AND predict on each of filtered signals
        # this should be 60 * number of seconds
        
        if self.config.debug_discrete_flag:
            print("classify_by_analogue_gesture:", self.win_downsample)
            
        decoded_velocities = np.zeros((2, self.win_downsample))
        
        # iteratively decoding  
        for ix in range( self.win_downsample - 1):
            
            filtered_signal = filtered_signals[:, ix]
            
            decoded_velocity = self.discrete_predict_velocity(filtered_signal)
            
            decoded_velocities[:,ix] = decoded_velocity.T
            

        decoded_cursor_position = np.sum(decoded_velocities, axis = 1) / self.win_downsample # get average decoded velocity
        # TODO: refactor self.state.decoded_velocity to **_position if we keep integration method
        self.state.disc_decoded_velocity = np.append(self.state.disc_decoded_velocity, decoded_cursor_position[:,None], axis = 1) # store decoded velocity
        # classify target from average velocity vector
        decoded_class = self.discrete_classify_by_target(decoded_cursor_position[:,None])

        if self.config.debug_discrete_flag:
            print("filtered_signal", np.max(filtered_signals))
            print("decoded velocity", decoded_velocity)

            print("decoded_cursor_position", decoded_cursor_position)
            print("decoded discrete class:", decoded_class)

        return int(decoded_class)



    def bound_decoder_output(self):
        """
        this function will access the self.state.decoded_position and self.state.decoded_velocity
        apply bounding to one or both of these variables
        """

        # access the varibles
        position = self.state.decoded_position[:, -1] #  
        velocity = self.state.decoded_velocity[:, -1]# appends v+
        

        # bound the variables (bound x/y separately)
        if abs(position[0]) > self.config.window_size_in_cm[0] / 2: # bound y position # 36 heuristically determined, based on screen size aspect ratio
            position[0] = self.state.decoded_position[0, -2] # keep at the last position that is within the bound. 
            velocity[0] = 0 # zero the velocity in the component at boundary
            self.state.bound_limit = self.state.bound_limit + 1
        
        if abs(position[1]) > self.config.window_size_in_cm[1] /2: # 24 is 'half screen height' set in top level graph #FIXME: set via config
            position[1] = self.state.decoded_position[1, -2]
            velocity[1] = 0
            self.state.bound_limit = self.state.bound_limit + 1

        if self.state.bound_limit > 100:
            position[0] = 0
            position[1] = 0
            velocity[0] = 0
            velocity[1] = 0
            self.state.bound_limit = 0

        # resave to the last update
        self.state.decoded_position[:, -1] = position
        self.state.decoded_velocity[:, -1] = velocity

    # collect batch data and calculated the intended velocity
    def learner(self):
        learner_ready = False

        self.calc_intended_velocity()

        # check to see if we can update the W and H matrices
        self.idx_batch += 1
        # check batch, update decoder transformation matrix at configured batch length
        if self.idx_batch > self.config.W_batch_length+1:
            self.idx_batch = 0
            learner_ready = True

        return learner_ready

    def calc_intended_velocity(self):
         # v_int = ref_pos[-1] - cursor_pos[-1])/|ref_pos[-1] - cursor_pos[-1]| --> gives the norm of intended velocity
        # TODO: gain may or may not be needed
        gain = 120
        intended_vector =(self.state.reference_position[:, -1] - self.state.decoded_position[:, -1])/ self.config.sampling_rate # TODO: is it decoded position[-1] or [-2]
        if np.linalg.norm(intended_vector) <= ALMOST_ZERO_TOL:
            intended_norm = np.zeros((2, 1))
        else:
            intended_norm = intended_vector * gain #/ np.linalg.norm(intended_vector) * gain
            intended_norm = np.reshape(intended_norm, (2, 1))

        # TODO: implement ring buffer to avoid appending?
        self.state.intended_velocity = np.append(self.state.intended_velocity,
                                                 intended_norm,
                                                 axis = 1)

    def assist(self):
        #TODO make the assist level linearly decreasing

        intended_velocity = self.state.intended_velocity[:,-1]

        fraction_prior = (1.0 - self.assist_level) * self.state.decoded_velocity[:, -1]
        assisted_fraction = self.assist_level * intended_velocity * self.assist_const

        self.state.decoded_velocity[:, -1] = fraction_prior + assisted_fraction

        # re-calculate the latest position
        position = self.state.decoded_position[:, -2]\
            + ( self.state.decoded_velocity[:, -1]/ self.config.sampling_rate)  # p+ = p + (v+)*dt
        #overwrite the latest decoded position
        self.state.decoded_position[:, -1] = np.squeeze(position)

    # set up gradient of cost:
    # d(c_L2(D))/d(D) = 2*(DF + HV - V+)*F.T + 2*lamdbaD*D
    def gradient_cost_l2(self, F, D, H, V, lambdaE, lambdaD, lambdaF):
        '''
        inputs: 
        F: 64 channels x time EMG signals
        V: 2 x time target velocity
        D: 2 (x y vel) x 64 channels decoder
        H: 2 x 2 state transition matrix

        outputs:
        grad_cost: returns the calculated gradient of the cost function (array)

        description:
        grad_cost = d(c_L2(D))/d(D) = 2*(DF + HV - V+)*F.T + 2*lamdbaD*D
        ''' 
        Nd = 2
        Ne = self.config.N_SENSORS
        
        D = np.reshape(D,(Nd, Ne))
        Vplus = V[:,1:]
        Vminus = V[:,:-1]

        g_e = 2 * (D@F + H@Vminus - Vplus) @ F.T * lambdaE
       
        g_d = 2 * lambdaD * D 

        grad = (g_e + g_d).flatten()

        return grad
 

    # set up the cost function: 
    # c_L2 = (||DF + HV - V+||_2)^2 + lambdaD*(||D||_2)^2 + lambdaF*(||F||_2)^2
    def cost_l2(self, F, D, H, V, lambdaE, lambdaD, lambdaF):
        '''
        inputs: 
        F: 64 channels x time EMG signals
        V: 2 x time target velocity
        D: 2 (x y vel) x 64 channels decoder
        H: 2 x 2 state transition matrix

        outputs:
        cost: returns the calculated cost function (scalar value)

        description:
        calculates and returns the set cost
        c_L2 = (||DF + HV - V+||_2)^2 + lambdaD*(||D||_2)^2 + lambdaF*(||F||_2)^2
        ''' 
        Nd = 2
        Ne = 64
        Nt = F.shape[-1]
        D = np.reshape(D, (Nd, Ne)) 
        F = np.reshape(F, (Ne, Nt))
        Vplus = V[:,1:]
        Vminus = V[:,:-1]
        
        c_e =  np.sum( (D @ F + H@Vminus - Vplus)**2 ) 
  
        c_d = np.sum( D**2 ) 

        c_f = np.sum( F**2 ) 

        cost = lambdaE * c_e  + lambdaD * c_d + lambdaF * c_f
        
        return cost

    def estimate_decoder(self, F, H, V):
        return (V[:,1:]-H@V[:,:-1])@np.linalg.pinv(F)

    def update(self):
        # grab batch data
        # grab sensor observations
        u = np.array(self.emg_ring_buffer[-(self.config.W_batch_length):]).T # needs to be num_dims x W_batch_length 
        
        # grab trial states
        q = self.state.intended_velocity[:, -(self.config.W_batch_length):] # needs to be 2 (velocity = x, y) x W_batch_length 

        # set the emg and velocity parameters
        F = copy.deepcopy(u[:, :-1])
        V = copy.deepcopy(q)

        # initial decoder estimate for gradient descent
        D0 = self.estimate_decoder(F, self.state.H, V)

        # set alphas
        alphaF=1e-2
        alphaD=1e-2

        # start time
        t0 = time.time()

        # use scipy minimize for gradient descent and provide pre-computed analytical gradient for speed
        out = minimize(lambda D: self.cost_l2(F,D,self.state.H,V, self.config.lambdaE, self.config.lambdaD, self.config.lambdaF), 
                                             D0,
                                             method='BFGS', 
                                             jac = lambda D: self.gradient_cost_l2(F,D,self.state.H,V, self.config.lambdaE, self.config.lambdaD, self.config.lambdaF), 
                                             options={'disp': True})

        # reshape to decoder parameters
        W_hat = np.reshape(out.x,(2, self.config.N_SENSORS))

        t1 = time.time()
        total = t1-t0
        print(total)

        # smoothbatch/low-filter smoothing update to new decoder
        W_new = self.config.alpha*self.state.W + ((1 - self.config.alpha) * W_hat)
        self.state.W_history = np.append(self.state.W_history, W_new[:, :, np.newaxis], axis = 2)
        self.state.W = W_new
        
        if len(self.state.reference_position[0,:]) > self.config.W_batch_length:
            print('error:', np.mean(self.state.reference_position[:,-(self.config.W_batch_length):]-self.state.decoded_position[:,-(self.config.W_batch_length):],axis=1))
    

    def discrete_predict_velocity(self, sensor_data):
        '''
        Args:
            sensor_data (np.ndarray): emg data within emg_window - corresponds to 'GO CUE' & used to predict/decode intent

        Returns:
            cursor_velocity (np.ndarray): predicted cursor velocity vector
        '''
        sensor_data_filter = sensor_data
        sensor_data_filter = np.squeeze(sensor_data_filter)[:, np.newaxis]
        # sensor_data_filter is size (64, 1024)
        
        # we need a separate W for discrete stuff
        velocity_sensor_update = np.squeeze(self.state.W_disc @ sensor_data_filter) 
        
        velocity = np.squeeze(velocity_sensor_update) # make sure the shapes match
        velocity = velocity[:,np.newaxis] 

        return velocity
    

    def discrete_classify_by_target(self, decoded_cursor_velocity, print_indx = False):
        # finish classification function for decoded_target
        # hint: use distance diff function. iterate through and find smallest distance
        # use smallest angle as classified target index
        
        scaled_cursor_velocity = self.scale_velocity_vector(decoded_cursor_velocity)
        
        # pick the smallest distance diff
        dist_diffs = np.linalg.norm((scaled_cursor_velocity.T - self.target_positions),
                                    axis = 1) # this should be an array
        
        min_dist_diffs = np.argmin(dist_diffs)

        classified_target = self.target_positions[min_dist_diffs,:]#[:, None]

        return min_dist_diffs

    
    def discrete_learner(self):

        if self.config.disc_decode_mode == "analogue_gesture":
            return self._learn_intended_vector()
        else:
            raise Exception(f"Discrete learner mode not implemented:", self.config.disc_decode_mode)
    
    def update_discrete_decoder(self):

        if self.config.disc_decode_mode == "analogue_gesture":
            self._update_disc_decoder_by_grad_descent()
        else:
            raise NotImplementedError(self.config.disc_decode_mode)

    
    def _learn_intended_vector(self):

        learner_ready = False

                # get the lastest cued target
        intended_vector = self.target_positions[self.state.current_reference_target_code_int, :]

        # TODO: implement ring buffer to avoid appending?
        if self.state.disc_intended_velocity.shape[0] != 2:
            raise Exception("_learn_intended_vector: disc_intended_velocity shape",  self.state.disc_intended_velocity.shape) 
        
        self.state.disc_intended_velocity = np.append(self.state.disc_intended_velocity,
                                                 intended_vector[:,np.newaxis],
                                                 axis = 1)
        self.disc_idx_batch += 1
        # check batch, update decoder transformation matrix at configured batch length
        if self.disc_idx_batch == self.config.learning_batch:
            self.disc_idx_batch = 0
            learner_ready = True

        return learner_ready
    
    def _update_disc_decoder_by_grad_descent(self):
        

        u = self.state.collected_emg_windows[:,-(self.config.learning_batch * self.num_win_downsamples):] # needs to be num_channels x num_timepoints x learning_batch

        # grab trial states
        # intended velocities (2 (x/y), trials)
        q_val = self.state.disc_intended_velocity[:, -(self.config.learning_batch):] # use cued positions as velocity vectors for updating decoder
        q = np.hstack([np.tile(q,(self.num_win_downsamples,1)).T for q in q_val.T])

        # emg_windows against intended_targets (trial specific cued target)
        F = copy.deepcopy(u[:,:-1]) # note: truncate F for estimate_decoder
        V = copy.deepcopy(q)
        

                # initial decoder estimate for gradient descent
        D0 = np.random.rand(2,self.config.N_SENSORS)#self.estimate_decoder(F, self.state.H, V)

        
        # set alphas
        alphaF=0
        alphaD=0

         # use scipy minimize for gradient descent and provide pre-computed analytical gradient for speed
         # TODO: self.state.H needs to be zeros
        out = minimize(lambda D: self.cost_l2(F,D,self.state.H,V, self.config.lambdaE, self.config.lambdaD, self.config.lambdaF), 
            D0, method='BFGS', 
            jac = lambda D: self.gradient_cost_l2(F,D,self.state.H,V, self.config.lambdaE, self.config.lambdaD, self.config.lambdaF), 
            options={'disp': True})

        # reshape to decoder parameters
        W_hat = np.reshape(out.x,(2, self.config.N_SENSORS))

        # smoothbatch/low-filter smoothing update to new decoder
        self.state.W_disc = self.config.alpha*self.state.W_disc+ ((1 - self.config.alpha) * W_hat)
        self.state.W_history_disc = np.append(self.state.W_history, 
                                             self.state.W_disc[:, :, np.newaxis], axis = 2)

    def scale_velocity_vector(self, velocity, radius = TARGET_LOCATION_RADIUS):
        """
        velocity: 2 by 1 vector
        """
        if np.linalg.norm(velocity) <= 1e-4:
            return np.array([0.0,0.0])

        unit_vector = velocity / np.linalg.norm(velocity)

        return unit_vector * radius

    
    def print_out_task_performance(self):

        if len(self.state.synced_reference_class) >= self.config.num_trials_for_counting_succusses:
            last_ref_classes = self.state.synced_reference_class[-self.config.num_trials_for_counting_succusses:]
            last_decoded_classes = self.state.decoded_disc_states[-self.config.num_trials_for_counting_succusses:]
            self.success_count = np.sum(last_ref_classes == last_decoded_classes)
            self.succuss_rate = self.success_count / self.config.num_trials_for_counting_succusses
            print("finished trials:", len(self.state.synced_reference_class),"number of trials in window:", self.config.num_trials_for_counting_succusses, "correct trials", self.success_count)
            print("succuss rate:",self.succuss_rate)
            




    
    # delinearize filter
    @classmethod
    def filt_delinearize(self, emg_raw):
        """
        inputs:
        emg_raw: the raw EMG signal to perform the RMS filtering on
        output:
        emg_filt: filtered EMG signal
        """
        # emg_raw = abs(np.asarray(emg_raw))
        wN = emg_raw.shape[1]
        # DETREND FIRST HALF OF DATA BEFORE TAKING AVERAGE (TODO: ask Momona about this)
        data1 = detrend(emg_raw[:,:int(wN/2)], axis=1,type='linear')
        data2 = detrend(emg_raw[:,int(wN/2):], axis=1,type='linear')
        datanew = np.concatenate((data1,data2),axis=1)
        emg_filt = np.mean(abs(datanew),axis=1)
        return emg_filt

    
    def _setup_hdf_save(self):
        """
        this function sets up a table expandable array where lastest task data is saved to
        """

        if self.config.hdf_save_path == "":
            self.hdf_save_path = os.getcwd()
            if self.config.verbose: 
                print(f'no file dir defined in the config file, use current dir {os.getcwd()} to save hdf file')
        else:
            self.hdf_save_path = self.config.hdf_save_path

        # set up file name 
        if not os.path.isdir(self.hdf_save_path):
            raise Exception(f'quattro hdf saving:  supplied dir {self.config.hdf_save_path} is not valid')
        
        hdf_save_full_path = os.path.join(self.hdf_save_path, 
                                          self.config.hdf_save_filename)
        print(f'check if its file {hdf_save_full_path}')

        if os.path.isfile(hdf_save_full_path):
            print(f'current file {self.config.hdf_save_filename} exists in {self.hdf_save_path}')
            now = datetime.datetime.now()
            self.hdf_save_filename = self.config.hdf_save_filename.rstrip('.h5') \
                                     + str(now.hour) + str(now.minute) + str(now.second)+ '.h5'
            hdf_save_full_path = os.path.join(self.hdf_save_path, 
                                          self.hdf_save_filename)
        else:
            self.hdf_save_filename = self.config.hdf_save_filename
            hdf_save_full_path = os.path.join(self.hdf_save_path, 
                                          self.hdf_save_filename)


        self._hdf_writer = hdfwriter.HDFWriter(hdf_save_full_path)
        #set up data type to save
        self._hdf_dt = np.dtype([('timestamp', np.float, (1,)),
                        ('decoded_position',np.float, (2,)), # two positions, x and y
                        ('decoded_velocity', np.float, (2,)), # two velocities, x and y
                        ('intended_velocity', np.float, (2,)), 
                        ('weiner_filter_w', np.float, (2, self.config.N_SENSORS)),
                        ('weiner_filter_h', np.float, (2, 2)),
                        ('raw_emg',np.float, (self.config.N_SENSORS, self.config.batch_length)),
                        ('filtered_emg', np.float, (self.config.N_SENSORS)), #TODO, is this the right dim? 
                        ('reference',np.float, (2,)),  # ref buffer's x,y directions.
                        ('discrete_state', f"<S{MAX_NUM_CHARACTERS_FOR_STATE_NAME}", (1,)),
                        ('disc_weiner_filter_w', np.float, (2, self.config.N_SENSORS)),
                        ('reference_discrete_code_int', np.int, (1,)),
                        ('filtered_signal_for_discrete',np.float, (self.config.N_SENSORS, self.num_win_downsamples)),
                        ('decoded_discrete_state', np.int, (1,)),
                        ('discrete_succuss_rate', np.float, (1,))
                        ]) # two velocities, x and y 

        self._hdf_writer.register(self.config.hdf_system_name, self._hdf_dt)

        if self.config.verbose:
            print(f'succussfully registered weiner hdf saving to {self.hdf_save_path + self.hdf_save_filename}')


    def _save_to_hdf_table(self, **kwargs):
        data = np.empty((1,),dtype = self._hdf_dt)

        for key, val in kwargs.items():
            if key in self._hdf_dt.fields.keys():
                data[key] = val
            else:
                raise Exception(f'supplied data {key} is not initialized in self._hdf_dt, will not get saved')
                
        self._hdf_writer.send(self.config.hdf_system_name, data)

    def _setup_nidaq_sync(self):
        """
        initialization functions to set up NIDAQ sync functionality.
        """
        from nidaq_sync import NidaqSync
        self._nidaq_sync = NidaqSync()
        self._nidaq_sync.setup_nidaq()
    
    def calculate_mouse_coordinate_2D(self, coordinate_2D):
        window_size =  np.array(self.config.window_size_in_pixels)[:, np.newaxis]
        coordinate_normalized = coordinate_2D / window_size - np.array([0.5, 0.5])

        # switch the coordinate
        coordinate_normalized = np.array([[1,0],[0,-1]]) @ coordinate_normalized

        # make this column vec so  compatible with coordinate_2D
        screen_cm = np.array(self.config.window_size_in_cm)[:, np.newaxis]
        coordinate_2D_physical = coordinate_normalized * screen_cm

        return coordinate_2D_physical

    
