import numpy as np
import labgraph as lg
from typing import Tuple, Dict

# inputs
from quattrocento_input import quattrocento_node, quattrocento_config
# from peripheral_inputs import MouseCoordinateGenerator, MouseGenConfig
from eye_inputs import MouseCoordinateGenerator, MouseGenConfig #added by Amber, eye tracker input

# reference target imports
from tracking_task import SigGenConfig, ReferenceGenerator

# decoders
from hybrid_decoder import HybridDecoderNode, HybridDecoderConfig #EMG decoder
from manual_hybrid_decoder import ManualHybridDecoderNode, ManualHybridDecoderConfig #added by Amber, eye tracker deocder (i.e. mouse cursor)

# graphics nodes
# from graphic_node import GraphicsNodeTask_2D_Decoder, GraphicsNodeConfig 
from graphic_node_eye import GraphicsNodeTask_2D_Decoder, GraphicsNodeConfig #added by Amber, one node with two cursor display

# ultility functions
from utility import calc_screen_size

LG_FUDGE_FACTOR_EXP = 1.0 / 0.8666 # experimentally determined
LG_FUDGE_FACTOR = 1.0

class ManualHybridTaskGraph(lg.Graph):
    ManualGraphicsNodeTracking: GraphicsNodeTask_2D_Decoder # where graphics come from
    Reference: ReferenceGenerator # where the reference -- trajectory tracking
    SensorStreamNode: MouseCoordinateGenerator # generates cursor
    ManualDecoderNode: ManualHybridDecoderNode # decoder

    def setup(self) -> None:
        
        # set up experiment
        subject_number = 42
        subject_folder = ""
        exp_length = 5.0 # minutes
        
        hybrid_mode = "color" # color, vertical_shift, or none
        one_dimension_tracking_flag = False
        
        screen_resolution = (1280, 720)
        SCREEN_HALF_HEIGHT = 24.0
        screen_size = calc_screen_size(screen_resolution, SCREEN_HALF_HEIGHT)
        
        # reference settings
        Default_color_code = 0
        REFERENCE_COLOR_CODES = (1,2)
        REFERENCE_LIST_CODE_TO_RGB = ((1,1,1,1), # 0
                                      (0,0,1,1), # 1
                                      (0,1,0,1)) # 2
        
        NUM_MOUSE_SENSORS = 3  # x,y and click
        MOUSE_SAMPLING_RATE = 60.0
        
        # directly pass out the x,y coordinates
        cont_decoder_initialization  = np.array([[1, 0, 0],
                                                 [0, 1, 0]], dtype=float)
        disc_decoder_initialization = np.array([0, 0, 1], dtype=float)

        self.SensorStreamNode.configure(
            MouseGenConfig(
                sampling_frequency=MOUSE_SAMPLING_RATE,
                left_button_clicked=True,
                debug_flag=True
            )
        )
        
        self.Reference.configure(
            SigGenConfig(
                discrete_property_mode=hybrid_mode, # color, vertical_pos, or None
                one_dim_flag=one_dimension_tracking_flag,
                reference_color_codes=REFERENCE_COLOR_CODES,
                change_disc_property_interval_in_second=4.0, #float             
                screen_cm=screen_size,
                scale_to_screen=True,
                sample_rate=60.0
            )
        )
        
        self.ManualDecoderNode.configure(
            ManualHybridDecoderConfig(
                total_num_cycles=int(60 * 60 * exp_length),  # 60 minutes * 60 seconds
                N_SENSORS=NUM_MOUSE_SENSORS,
                sensor_packet_rate=MOUSE_SAMPLING_RATE,
                filter_mode="pass_through",
                cont_decoder_mode="pos_pass_through",
                batch_length=1, # number of data points to grab for continuous action
                decoder_adaptable=False,
                default_discrete_class=0,
                initial_cont_decoder=cont_decoder_initialization,
                discrete_decode_window_seconds_after_property_change= 1.0,
                discrete_batch_length=60, # number of data points to grab for decode discrete action
                initial_disc_decoder=disc_decoder_initialization,
                subject_id=subject_number,
                decoder_id=1,
                hdf_save_path=subject_folder,
                hdf_save_filename='hybrid_mouse_control_weiner_task_data_METAS_0' + str(subject_number) + '.h5',
                window_size_in_pixels=screen_resolution,
                window_size_in_cm=screen_size,
                disc_decode_mode = "constant_class" # added by Amber
            )
        )

        self.ManualGraphicsNodeTracking.configure(
            GraphicsNodeConfig(
                screen_size=screen_size,
                screen_half_height=SCREEN_HALF_HEIGHT,
                framerate=60.0,
                test_monitor_resolution=screen_resolution,
                reference_list_code_to_rgb=REFERENCE_LIST_CODE_TO_RGB 
            )
        )
    
    def connections(self) -> lg.Connections:
        return(
            (self.SensorStreamNode.OUTPUT, self.ManualDecoderNode.SENSOR_INPUT),
            (self.Reference.OUTPUT, self.ManualDecoderNode.REFERENCE),
            (self.ManualDecoderNode.CONTROL_OUTPUT, self.ManualGraphicsNodeTracking.CURSOR_INPUT2),
            (self.ManualDecoderNode.REFERENCE_OUTPUT, self.ManualGraphicsNodeTracking.REFERENCE_INPUT)
        )

    
    def process_modules(self) -> Tuple[lg.Module, ...]:
        return (
                (self.SensorStreamNode,
                 self.Reference,
                 self.ManualDecoderNode,
                 self.ManualGraphicsNodeTracking)
            )



decoder_number = 8 # int that references which decoder was used
block_number = 0

learning_rate = 0.5
save_learning = 75

# penalty parameter
lambdaE = 1e-6
lambdaF = 1e-7
# higher penalty
if decoder_number == 1 or decoder_number == 3 or decoder_number == 5 or decoder_number == 7:
    # learning rate
    lambdaD = 1e-3
# lower penalty
elif decoder_number == 2 or decoder_number == 4 or decoder_number == 6 or decoder_number == 8:
    lambdaD = 1e-4


# Define Decoder Calibration Graph
class EmgHybridTaskGraph(lg.Graph):
    GraphicsNodeTracking: GraphicsNodeTask_2D_Decoder #graphic node with two cursor displays
    Reference: ReferenceGenerator
    EMGStreamNode: quattrocento_node # generates EMG cursor
    ManualStreamNode: MouseCoordinateGenerator # generates manual/eye tracker cursor
    DecoderNode: HybridDecoderNode # EMG decoder
    ManualDecoderNode: ManualHybridDecoderNode #manual/eye tracker decoder (simply return xy coordinates)

    def setup(self) -> None:

        screen_resolution = (1900, 1120)
        SCREEN_HALF_HEIGHT = 24.1
        # screen_resolution = (1280, 720)
        # SCREEN_HALF_HEIGHT = 24.0

        screen_size = calc_screen_size(screen_resolution, SCREEN_HALF_HEIGHT)
        print("screen size:", screen_size)

        # set up experiment
        subject_number = 101 # keep this fixed on brl
        subject_folder = ""
        exp_length = 5.0 # minutes

        # sensor seettings
        NUM_QUOTTRO_CHANNELS = 64
        QUOTTRO_PACKET_RATE = 32.0

        # reference settings
        Default_color_code = 0
        REFERENCE_COLOR_CODES = (1,2)
        REFERENCE_LIST_CODE_TO_RGB = ((1,1,1,1), # 0
                                      (0,0,1,1), # 1
                                      (0,1,0,1)) # 2

        NUM_MOUSE_SENSORS = 3  # x,y and click
        MOUSE_SAMPLING_RATE = 60.0

        hybrid_mode = "color" # color, vertical_shift, or none
        one_dimension_tracking_flag = True

        # decoder settings
        # data_path = 'D:\\Momona\\meta_emg\\20220510_hybrid_development' # for BRL desktop
        # dec_init_path = data_path + '\\decoder_initializations\\'
        # decoder_initialization = np.load(dec_init_path + 'METACPHS_S' + str(subject_number) + '_dneg.npz')['arr_0'][0]
    
        # discrete decoder
        decoder_initialization = np.random.random((2,64)) * (1e-3)
        disc_decoder_initialization = np.zeros((2,64))

        # directly pass out the x,y coordinates
        cont_decoder_initialization  = np.array([[1, 0, 0],
                                                 [0, 1, 0]], dtype=float)
        
        self.EMGStreamNode.configure(
            quattrocento_config(
                use_dummy_quattro_inputs = False,
                sampling_frequency = 2048, 
                num_data_points= 30,
                decoder_initialization = decoder_number,          
                # set path and filename for saving   
                #hdf_save_path = data_path,
                hdf_save_filename  = 'quattro_data_METACPHS_S' + str(subject_number) \
                    + '_L2_a_' + str(save_learning) + '_D_' + str(decoder_number)  + '_BLOCK_' + str(block_number)  \
                    + '_pE_' + '%.0e' %lambdaE + '_pD_' + '%.0e' %lambdaD + '_pF_' + '%.0e' %lambdaF + '.h5'
                )  
        )

        self.ManualStreamNode.configure(
            MouseGenConfig(
                sampling_frequency=MOUSE_SAMPLING_RATE,
                left_button_clicked=False,
                debug_flag=True
            )
        )
        
        self.Reference.configure(
            SigGenConfig(
                discrete_property_mode=hybrid_mode, # color, vertical_pos, or None
                one_dim_flag=one_dimension_tracking_flag,
                initial_ref_y_position=0.5, #relative to the screeen
                reference_color_codes=REFERENCE_COLOR_CODES,
                change_disc_property_interval_in_second=4.0, #float             
                screen_cm=screen_size,
                scale_to_screen=True,
                sample_rate=60.0
            )
        )

        self.GraphicsNodeTracking.configure(
            GraphicsNodeConfig(
                screen_size=screen_size,
                screen_half_height=SCREEN_HALF_HEIGHT,
                framerate=60.0,
                test_monitor_resolution=screen_resolution,
                reference_list_code_to_rgb=REFERENCE_LIST_CODE_TO_RGB 
            )
        )

        self.DecoderNode.configure(
            HybridDecoderConfig(
                subject_id=subject_number,
                one_d_task_flag = one_dimension_tracking_flag,
                cycle_rate = 60.0,
                sampling_rate = 60.0,
                total_num_cycles=int(60 * 60.0 * exp_length),  # 60 minutes * 60 seconds
                N_SENSORS=NUM_QUOTTRO_CHANNELS,
                sensor_packet_rate=QUOTTRO_PACKET_RATE,
                filter_mode="filter_emg",
                enable_cont_decoder =  True,
                cont_decoder_mode="velocity_based_weiner",
                decoder_id=1,
                batch_length=204, # number of data points to grab for continuous action
                decoder_adaptable=True,
                initial_cont_decoder=decoder_initialization,
                save_final_decoder = False, 
                alpha=learning_rate,
                lambdaE=lambdaE,
                lambdaD=lambdaD,
                lambdaF=lambdaF,
                enable_discrete_decoder = True,
                disc_decode_mode = "analogue_gesture",
                discrete_decode_window_seconds_after_property_change= 1.0,
                discrete_batch_length=820, # number of data points to grab for decode discrete action
                discrete_decoder_adaptable = True,
                initial_disc_decoder=disc_decoder_initialization,
                hdf_save_path=subject_folder,
                hdf_save_filename='weiner_task_data_METAS_0' + str(subject_number) + '.h5',
                window_size_in_pixels=screen_resolution,
                window_size_in_cm=screen_size
            )
        )

        self.ManualDecoderNode.configure(
            ManualHybridDecoderConfig(
                total_num_cycles=int(60 * 60 * exp_length),  # 60 minutes * 60 seconds
                N_SENSORS=NUM_MOUSE_SENSORS,
                sensor_packet_rate=MOUSE_SAMPLING_RATE,
                filter_mode="pass_through",
                cont_decoder_mode="pos_pass_through",
                batch_length=1, # number of data points to grab for continuous action
                decoder_adaptable=False,
                default_discrete_class=0,
                initial_cont_decoder=cont_decoder_initialization,
                discrete_decode_window_seconds_after_property_change= 1.0,
                discrete_batch_length=60, # number of data points to grab for decode discrete action
                initial_disc_decoder=disc_decoder_initialization,
                subject_id=subject_number,
                decoder_id=1,
                hdf_save_path=subject_folder,
                hdf_save_filename='hybrid_mouse_control_weiner_task_data_METAS_0' + str(subject_number) + '.h5',
                window_size_in_pixels=screen_resolution,
                window_size_in_cm=screen_size,
                disc_decode_mode = "constant_class" # added by Amber
            )
        )

            
    def connections(self) -> lg.Connections:
        return(
            (self.EMGStreamNode.OUTPUT, self.DecoderNode.SENSOR_INPUT), # EMG to EMG Decoder
            (self.ManualStreamNode.OUTPUT, self.ManualDecoderNode.SENSOR_INPUT), # Manual Input to Manual Decoder
            (self.Reference.OUTPUT, self.DecoderNode.REFERENCE), # Reference Task to Decoder Reference
            (self.DecoderNode.CONTROL_OUTPUT, self.GraphicsNodeTracking.CURSOR_INPUT), # EMG Decoder Output to Graphics (i.e. display cursor)
            (self.DecoderNode.REFERENCE_OUTPUT, self.GraphicsNodeTracking.REFERENCE_INPUT), # Decoder Reference Output to Graphics (i.e. display target)
            (self.ManualDecoderNode.CONTROL_OUTPUT, self.GraphicsNodeTracking.CURSOR_INPUT2) # Manual Decoder Output to Graphics (i.e. display mouse cursor)
        )

    #everything would run in a single process. In this way, process_modules() lets us concisely specify how a graph should be parallelized.
    def process_modules(self) -> Tuple[lg.Module, ...]:
        return ((self.EMGStreamNode,
                self.ManualStreamNode,
                self.Reference,
                self.DecoderNode,
                self.ManualDecoderNode,
                self.GraphicsNodeTracking
        ))

if __name__ == "__main__":
    # Run Decoder Calibration
    # lg.run(ManualHybridTaskGraph)
    lg.run(EmgHybridTaskGraph)