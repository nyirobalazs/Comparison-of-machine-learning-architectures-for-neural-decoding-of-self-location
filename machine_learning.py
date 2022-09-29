from doctest import testsource
import pathlib
from sklearn.metrics import r2_score
from pydoc import describe
import numpy as np
import tensorflow as tf
from tensorflow import keras, random
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
#from keras.layers import CuDNNLSTM #--> TODO: try as a faster GPU option for LSTM layers 
from keras.models import Model
#from keras import layers
from tensorflow.keras import layers, regularizers
import keras.backend as K
from keras.layers.core import *
from keras.callbacks import ModelCheckpoint
import copy
from sklearn.model_selection import train_test_split
import pandas as pd
import operator as op
import read_write as rw
from read_write import DataFile,print_terminal
import preprocess as pr
import torch
print(print_terminal(type='done',message='Python package initialization done in machine_learning.'))
   
if torch.cuda.is_available():
    print(print_terminal(type='run',message='Num GPUs Available for Tensorflow:{}').format(len(tf.config.list_physical_devices('GPU'))))


#---------MODEL GENERATING TOOLS#---------
def multify_weights(kernel, out_channels):
    """
    Calculate mean for the extra weight dimensions
    ----------
    Args:
        kernel (array): kernel weights
        out_channels (int): number of channels in the output kernel
    """
    mean_1d = np.mean(kernel, axis=-2).reshape(kernel[:,:,-1:,:].shape)
    tiled = np.tile(mean_1d, (out_channels, 1))
    return(tiled)

#TODO: input kernelt beadni inputnak nem a hidden layerts
def pad_with_calc(start_dim, end_dim, to_end=True):
    """
    Calculate the padwith tuples for numpy pad function.
    ----------
    Args:
        start_dim (tuple): dimension of the original kernel weight matrix
        end_dim (tuple): dimension of the new kernel weight matrix
        to_end (bool, optional): If it's True the pad function will add the new 
        mean values to the end of the weight matrixes.Otherwise to the begining.
        Defaults to True.

    Returns:
        pad_with(array of tuples): the pad_with array to get the required dimension output with numpy pad func.
    """
    
    if len(start_dim)!=len(end_dim):
        print(print_terminal(type='error',message='Lengths of dimensions are not equal in pad_with_calc function.'))
        raise ValueError('Lengths of dimensions are not equal in pad_with_calc function.')
    else:
        pad_with = []
        for _ in range(len(start_dim)):
            diff = end_dim[_]-start_dim[_]
            if to_end:
                pad_with.append((0,diff))
            else:
                pad_with.append((diff,0))
    return pad_with

def weightify(model_orig, custom_model, layer_modify):

  """
  Extend the original transfer learning model weights to the required number of channels, and updates the custom_model network.
  ----------
  Args:
    model_orig (tensorflow model obj): the original network model with the pretrained weights.
    custom model (tensorflow model obj): the network model with the required numbers of channels.
    layer_modify (str): the name of the first convolutional layer.
  """

  layer_to_modify = [layer_modify]

  conf = custom_model.get_config() #get config dict
  layer_names = [conf['layers'][x]['name'] for x in range(len(conf['layers']))] #list with the name of the layers
  
  for layer in model_orig.layers:
    if layer.name in layer_names:
      if layer.get_weights() != []: #if the layer has weights
        target_layer = custom_model.get_layer(layer.name)

        if layer.name in layer_to_modify:  
          kernels = layer.get_weights()[0]
          biases  = layer.get_weights()[1]

          kernels_extra_channel = np.concatenate((kernels,
                                                  multify_weights(kernels, 0)),
                                                  axis=-2) # For channels_last
                                                  
          target_layer.set_weights([kernels_extra_channel, biases])
          target_layer.trainable = False

        else:
          try:
            target_layer.set_weights(layer.get_weights()) #try to copy the weights to the new model
            target_layer.trainable = False
          except: #if it fails do to dimnsion issues
            if (np.array(layer.get_weights()[-1])==0).all():
              dim_target_layer = np.array(target_layer.get_weights()[:-1]).shape
              dim_layer = np.array(layer.get_weights()[:-1]).shape
              extr_target = np.pad(layer.get_weights()[:-1],pad_with_calc(dim_layer,dim_target_layer),'mean')
              out = []
              for _ in range(len(extr_target)):
                out.append(extr_target[_])
              out.append(layer.get_weights()[-1])
              target_layer.set_weights(out)
            else:
              out = []
              dim_layer = np.concatenate(np.array(layer.get_weights())).shape
              dim_target_layer = np.concatenate(np.array(target_layer.get_weights())).shape
              input = np.concatenate(np.array(layer.get_weights()))
              extr_target = np.pad(input,pad_with_calc(dim_layer,dim_target_layer),'mean')
              target_layer.set_weights(list([extr_target]))
            target_layer.trainable = False

#-----------------MODELS------------------
class CNN_BiLSTM_model:
    
    def attention_block(self, inputs, single_attention_vector=False):
        # inputs_shape-->(batch_size, time_steps, input_dim)
        
        time_steps = K.int_shape(inputs)[1]
        input_dim = K.int_shape(inputs)[2]
        a = layers.Permute((2, 1))(inputs)
        a = layers.Dense(time_steps, 
                         kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), 
                         bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(a)
        if single_attention_vector:
            a = Lambda(lambda x: K.mean(x, axis=1))(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((2, 1))(a)
        output_attention_mul = layers.Multiply()([inputs, a_probs])
        
        return output_attention_mul
    
    def make_default_hidden_layers(self, input_kernel, num_lstm_units):
        
        x = layers.Conv1D(filters = 64, kernel_size = 1, activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(input_kernel)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.LSTM(num_lstm_units, 
                                             return_sequences=True,
                                             kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), 
                                             recurrent_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                                             bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))(x)
        x = layers.TimeDistributed(Dense(1))(x)
        x = layers.Dropout(0.3)(x)
        x = self.attention_block(x)
        x = layers.Flatten()(x)
    
        return x
   
    def build_speed_branch(self, input_kernel, nlstm_units=500):   

        hidden_layer = self.make_default_hidden_layers(input_kernel=input_kernel, num_lstm_units=nlstm_units)
        x = layers.Dense(1,  
                         kernel_regularizer=regularizers.l2(0.01), 
                         bias_regularizer=regularizers.l2(0.01),
                         activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(hidden_layer)
        x = layers.Activation("sigmoid", name="speed_output")(x)
        return x
    
    def build_head_dir_branch(self, input_kernel, nlstm_units=500):   

        hidden_layer = self.make_default_hidden_layers(input_kernel=input_kernel, num_lstm_units=nlstm_units)
        x = layers.Dense(1, 
                         kernel_regularizer=regularizers.l2(0.01), 
                         bias_regularizer=regularizers.l2(0.01),
                         activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(hidden_layer)
        x = layers.Activation("sigmoid", name="head_dir_output")(x)
        return x
    
    def build_pos_X_branch(self, input_kernel, nlstm_units=500):   

        hidden_layer = self.make_default_hidden_layers(input_kernel=input_kernel, num_lstm_units=nlstm_units)
        x = layers.Dense(1,  
                         kernel_regularizer=regularizers.l2(0.01), 
                         bias_regularizer=regularizers.l2(0.01),
                         activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(hidden_layer)
        x = layers.Activation("sigmoid", name="position_X_output")(x)
        return x
    
    def build_pos_Y_branch(self, input_kernel, nlstm_units=500):   

        hidden_layer = self.make_default_hidden_layers(input_kernel=input_kernel, num_lstm_units=nlstm_units)
        x = layers.Dense(1,  
                         kernel_regularizer=regularizers.l2(0.01), 
                         bias_regularizer=regularizers.l2(0.01),
                         activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(hidden_layer)
        x = layers.Activation("sigmoid", name="position_Y_output")(x)
        return x    
    
    def assemble_full_model(self, input_shape, nlstm_units=500):
        #input_shape-->(time_steps, nchannels)
        
        input_kernel = layers.Input(shape=input_shape)
        speed_branch = self.build_speed_branch(input_kernel, nlstm_units)
        head_dir_branch = self.build_head_dir_branch(input_kernel, nlstm_units)
        pos_x_branch = self.build_pos_X_branch(input_kernel, nlstm_units)
        pos_y_branch = self.build_pos_Y_branch(input_kernel, nlstm_units)
        model = Model(inputs=input_kernel,
                     outputs = [speed_branch, head_dir_branch, pos_x_branch, pos_y_branch],
                     name="CNN_BiLSTM_net")
        
        print(print_terminal(type='done',message='CNN-BiLSTM modell assembling done with input shape {}.'.format(input_shape)))
        
        return model

class CNN_transferlearn_model:
    """
    The transfer learning model.
    """
    
    def make_default_hidden_layers(self, input_kernel, input_size):

        model = EfficientNetB0(include_top=False, weights="imagenet")
        config = model.get_config()

        model_custom =  EfficientNetB0(include_top=False,input_tensor=input_kernel, weights=None)
        modify_name = config["layers"][12]["config"]["name"]

        weightify(model, model_custom, modify_name)
        x = layers.GlobalAveragePooling2D(name="avg_pool")(model_custom.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.8, name="top_dropout")(x)

        return x

    def build_speed_branch(self, hidden_layer):
        
        x = layers.Dense(1, name="speed_output",  
                         kernel_regularizer=regularizers.l2(0.1), 
                         bias_regularizer=regularizers.l2(0.1),
                         activity_regularizer=regularizers.l1_l2(l1=0.1, l2=0.1))(hidden_layer)

        
        return x
    
    def build_head_dir_branch(self, hidden_layer):
        
        x = layers.Dense(1, name="head_dir_output",  
                         kernel_regularizer=regularizers.l2(0.1), 
                         bias_regularizer=regularizers.l2(0.1),
                         activity_regularizer=regularizers.l1_l2(l1=0.1, l2=0.1))(hidden_layer)

        return x
    
    def build_pos_X_branch(self, hidden_layer):   

        x = layers.Dense(1, name="position_X_output", 
                         kernel_regularizer=regularizers.l2(0.1), 
                         bias_regularizer=regularizers.l2(0.1),
                         activity_regularizer=regularizers.l1_l2(l1=0.1, l2=0.1))(hidden_layer)

        
        return x
    
    def build_pos_Y_branch(self, hidden_layer):   

        x = layers.Dense(1, name="position_Y_output", 
                         kernel_regularizer=regularizers.l2(0.1), 
                         bias_regularizer=regularizers.l2(0.1),
                         activity_regularizer=regularizers.l1_l2(l1=0.1, l2=0.1))(hidden_layer)

        
        return x
    
    def assemble_full_model(self, input_shape):

        input_kernel = layers.Input(shape=input_shape)
        inputs = self.make_default_hidden_layers(input_kernel=input_kernel,input_size=input_shape)
        speed_branch = self.build_speed_branch(inputs)
        head_dir_branch = self.build_head_dir_branch(inputs)
        pos_x_branch = self.build_pos_X_branch(inputs)
        pos_y_branch = self.build_pos_Y_branch(inputs)
        model = Model(inputs=input_kernel,
                     outputs = [speed_branch, head_dir_branch, pos_x_branch, pos_y_branch],
                     name="transferlearn_net")
        
        print(print_terminal(type='done',message='Transfer learning modell assembling done with input shape {}.'.format(input_shape)))
        return model

class CNN_behav_cloning_model:
    """
    The behaviour cloning model.
    """
    
    def make_default_hidden_layers(self, inputs):
        """
        Create hidden layer structure.
        ----------
        Args:
            inputs (input layer): The input layer with the required input dimensions.

        Returns:
            x(tf model): The model structure with the hidden layers.
        """

        x = layers.Conv2D(16, (3, 3), padding="same", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(inputs)
        x = layers.Activation("selu")(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)
        x = layers.Activation("selu")(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)
        x = layers.Activation("selu")(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.3)(x)
        return x

    def build_speed_branch(self, inputs):
        """
        Creates the output banch for the speed values.
        ----------
        Args:
            inputs(tf model): hidden layer arthitecture.
        Returns:
            x(tf model): speed banch output.
        """   

        x = self.make_default_hidden_layers(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(128, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)
        x = layers.Activation("selu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1, 
                         kernel_regularizer=regularizers.l2(0.01), 
                         bias_regularizer=regularizers.l2(0.01),
                         activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)
        x = layers.Activation("sigmoid", name="speed_output")(x)
        
        return x
    
    def build_head_dir_branch(self, inputs):  
        """
        Creates the output banch for the head direction values.
        ----------
        Args:
            inputs(tf model): hidden layer arthitecture.
        Returns:
            x(tf model): head dir banch output.
        """    

        x = self.make_default_hidden_layers(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(128, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)
        x = layers.Activation("selu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        #x = layers.Dense(1, name="head_dir_output")(x)
        x = layers.Dense(1, 
                         kernel_regularizer=regularizers.l2(0.01), 
                         bias_regularizer=regularizers.l2(0.01),
                         activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)
        x = layers.Activation("sigmoid", name="head_dir_output")(x)
        
        return x
    
    def build_pos_X_branch(self, inputs):   
        """
        Creates the output banch for the x values of the animal's position.
        ----------
        Args:
            inputs(tf model): hidden layer arthitecture.
        Returns:
            x(tf model): x coordinate banch output.
        """   

        x = self.make_default_hidden_layers(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(128, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)
        x = layers.Activation("selu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        #x = layers.Dense(1, name="position_X_output")(x)
        x = layers.Dense(1, 
                         kernel_regularizer=regularizers.l2(0.01), 
                         bias_regularizer=regularizers.l2(0.01),
                         activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)
        x = layers.Activation("linear", name="position_X_output")(x)
        
        return x
    
    def build_pos_Y_branch(self, inputs):   
        """
        Creates the output banch for the y values of the animal's position.
        ----------
        Args:
            inputs(tf model): hidden layer arthitecture.
        Returns:
            x(tf model): y coordinate banch output.
        """   

        x = self.make_default_hidden_layers(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(128, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)
        x = layers.Activation("selu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        #x = layers.Dense(1, name="position_Y_output")(x)
        x = layers.Dense(1, 
                         kernel_regularizer=regularizers.l2(0.01), 
                         bias_regularizer=regularizers.l2(0.01),
                         activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)
        x = layers.Activation("linear", name="position_Y_output")(x) #position_output
        
        return x
    

    def assemble_full_model(self, input_shape):
        """
        It creates the full model of the behaviour cloning network.
        ----------
        Args:
            input_shape (tuple): the sizes of the input datas.
        Returns:
            model(tf model): the whole model arthitecture
        """

        inputs = layers.Input(shape=input_shape)
        speed_branch = self.build_speed_branch(inputs)
        head_dir_branch = self.build_head_dir_branch(inputs)
        pos_x_branch = self.build_pos_X_branch(inputs)
        pos_y_branch = self.build_pos_Y_branch(inputs)
        model = Model(inputs=inputs,
                     outputs = [speed_branch, head_dir_branch, pos_x_branch, pos_y_branch],
                     name="behav_cloning_net")
        
        print(print_terminal(type='done',message='Behaviour cloning modell assembling done with input shape {}.'.format(input_shape)))
        return model

#-------------SET UP TRAINING-------------
class Training:
    
    def __init__(self,
                 window_size, 
                 init_lr=1e-2, 
                 epochs=100, 
                 batch_size=128, 
                 valid_batch_size=200,
                 decay_steps= 10,
                 live_stream=False,
                 dataset=None,
                 chunk_size=None,
                 is_gpu=None,
                 is_paral=None,
                 cutoff=None,
                 tasks=None,
                 is_save=True,
                 use_saver=True):
        
        self.init_lr          = init_lr
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.valid_batch_size = valid_batch_size
        self.decay_steps      = decay_steps
        self.live_steam       = live_stream
        self.window_size      = window_size
        self.tasks            = tasks
        self.is_save          = is_save
        self.use_saver        = use_saver
        
        if live_stream:
            if (dataset==None or chunk_size==None or is_paral==None or is_gpu==None or is_paral==None or cutoff==None or tasks==None):
                print(print_terminal(type='error',message='Missing argument(s) at Training init function, when live_stream is True.'))
                raise ValueError('Missing argument(s) at Training init function, when live_stream is True.')
                
        self.prep = pr.Preprocess(dataset,
                                window_size= window_size,
                                chunk_size = chunk_size,
                                is_gpu     = is_gpu,
                                is_paral   = is_paral,
                                cutoff     = cutoff,
                                ml_in_use  = True,
                                use_saver  = self.use_saver)

    def filter(self, df,column, operator, val):
        """
        It returns with those dataframe  and window indexes where the required 
        parameter(eg.: speed) is fits to the required parameters.
        ----------
        Args:
            df (pandas dataframe): the dataframe with the speed,head direction and position values.
            column (str): the name of that column which required to be filtered.
            operator (str): the operator for filtering(eg.: >,<,==)
            val (int): the value that describes the right data(eg.: data point should be > than val)

        Returns:
            idx(array): arrayes of indexes where the datapoints fit to the required parameters.
            filt_df(pandas dataframe): the filered dataframe.
        """
        
        opers = {'<':op.lt,'<=':op.le,'==': op.eq,'!=': op.ne,'>=':op.ge,'>':op.gt}

        if (column!=None or operator!=None or val!=None):
            if opers.get(operator)==None: raise ValueError('Operator name is not valid!')
            if not column in df: raise ValueError('Column name is not valid!')
            idx     = df[opers.get(operator)(df[column],float(val))].index
            filt_df = df[opers.get(operator)(df[column],float(val))]
        else:
            print(print_terminal(type='error',message='Missing paramater at filter.'))
            raise ValueError('Missing paramater at filter.')
        
        print(print_terminal(type='done',message='Dataset filtering done.'))
        return idx, filt_df


    def split_dataset(self, dataset,valid_ratio=0.3,test_ratio=0.1,shuffle=True,column=None,operator=None,val=None):
        """
        Split dataset into training, test and validation parts and also shuffle the dataset if it's required.
        ----------
        Args:
            dataset (obj): The dataset of the preprocessed signal
            valid_ratio (float, optional): ratio of the validation dataset compared to the whole dataset. Defaults to 0.3.
            test_ratio (float, optional): ratio of the test dataset compared to the whole dataset. Defaults to 0.1.
            shuffle (bool, optional): If it's True, dataset will be shuffled. Defaults to True.
            column (str, optional): name of the column which needed to be filtered. Defaults to None.
            operator (str, optional): Given column's datapoints rel. Defaults to None.
            val (int): the value that describes the right data(eg.: data point should be > than val). Defaults to None.
        Returns:
            train_ind, valid_ind, test_ind(array): indexes of the train, validation and test datasets
        """
        
        if self.live_steam and self.window_size==None: raise ValueError('Wrong window size at split_dataset')
        dataset_size = dataset.processed_data.shape[0]
        
        #Calculate head_dir,speed and position parameters from LED positions
        id_hd,id_speed,id_pos = self.prep.convert_LEDs(dataset.tetrode_srate, 
                                                        dataset.tetrode_timestamps, 
                                                        dataset.led_timestamps, 
                                                        dataset.raw_led_positions)
        #Create datarame
        self.y = pd.DataFrame({'speed': id_speed,
                            'head_dir': id_hd,
                            'position_x': id_pos[:,0],
                            'position_y': id_pos[:,1]  
                            })
        
        self.max_values = self.y.abs().max()
        self.y = self.y/self.max_values
        
        #filter dataframe
        if (column!=None and operator!=None and val!=None):
            ind_order, *_ = filter(self.y,column, operator, val)
            ind_order = ind_order[0:dataset_size]
        else:
            ind_order = np.array(self.y.index)[0:dataset_size]
        
        #shuffle if it shuffle=True
        if shuffle:
            ind_order = np.array(random.shuffle(ind_order))
        
        #Split dataset into parts in two steps
        n_valid = len(ind_order)*valid_ratio
        main_ind, test_ind = train_test_split(ind_order, test_size= test_ratio)
        re_valid_ratio = n_valid/len(main_ind)
        train_ind, valid_ind = train_test_split(main_ind, test_size= re_valid_ratio)
        
        return train_ind, valid_ind, test_ind 
        
        
    def generate_data(self, dataset, data_idx, is_training,channels=None, stype = 'ft_extr', batch_size=16):
        """
        Data generator will feed data during the training. 
        If: 
            self.live_steam=False --> It gets the preprocessed dataset and feed it into the learning func.
            self.live_steam=True  --> It will generate the data from the raw data during the process. 
                                      This function saves memory.
        ----------
        Args:
            dataset (obj): The dataset of the preprocessed signal
            data_idx (array): Indexes from split_dataset function
            is_training (bool): if it's False --> only on batch will be generated.
            stype (str, optional): the type of the processed signal which needed to be used. Defaults to 'ft_extr'.
                                   Options: 'ft_extr' -- 3D wavelet transfomed data || 'cont'-- processed 2D signal.
            batch_size (int, optional): Size of one batch. Defaults to 16.

        Yields:
            signals, speeds, head_dirs, position_X, position_Y (array): the signal and the decoded parameters of the windows.
        """
    
        # arrays to store our batched data
        signals, speeds, head_dirs, position_X, position_Y = [], [], [], [], []
        while True:
            for idx in data_idx:
                
                #split the given row of the dataframe into decded parameters
                df_line  = self.y.iloc[idx]
                speed    = df_line['speed']
                head_dir = df_line['head_dir']
                pos_X    = df_line['position_x']
                pos_Y    = df_line['position_y']

                if not self.live_steam: #if working from preprocessed dataset
                    #choose data type
                    if stype == 'ft_extr':
                        signal   = np.array(dataset.ft_extracted[idx])
                    elif stype == 'cont':
                        signal   = np.array(dataset.processed_data[idx])
                    else:
                        raise ValueError('Unknown stype at generate_data.')
                else: #if producing preprocessed data with size of a window, during the learning process
                    wind_start  = idx*self.window_size
                    wind_stop   = wind_start+self.window_size
                    raw_chunk   = np.array(dataset.raw_signal[wind_start:wind_stop,:])
                    time_chunk  = np.array(dataset.tetrode_timestamps[wind_start:wind_stop])
                    
                    #preprocess of the given window
                    (ft_extr, 
                     cont_signal, 
                     timestamps)=self.prep.process_window(input_signal=raw_chunk, 
                                                        timestamps_out=time_chunk, 
                                                        tasks=self.tasks, 
                                                        srate=dataset.tetrode_srate,
                                                        channels=channels,
                                                        dim_check=False)
                    #choose the output signal type
                    if stype == 'ft_extr':
                        signal   = ft_extr
                    elif stype == 'cont':
                        signal   = cont_signal
                
                speeds.append(speed)
                head_dirs.append(head_dir)
                position_X.append(pos_X)
                position_Y.append(pos_Y)
                smax=np.max(signal)
                signal=signal/smax
                signals.append(signal)
                
                # yielding condition
                if len(signals) >= batch_size:
                    yield signals, [speeds, head_dirs, position_X, position_Y]
                    signals, speeds, head_dirs, position_X, position_Y = [], [], [], [], []
                    
            if not is_training:
                break
    
    def euclidean_loss(self, y_true, y_pred):
        res = tf.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
        return res
    
    def cyclical_mae_rad(self, y_true, y_pred):
        return K.mean(K.minimum(K.abs(y_pred - y_true), K.minimum(K.abs(y_pred - y_true + 2*np.pi), K.abs(y_pred - y_true - 2*np.pi))), axis=-1)
    
    def mse(self, y_true, y_pred):
        return tf.keras.losses.MSE(y_true, y_pred)

    def mae(self, y_true, y_pred):
        return tf.keras.losses.MAE(y_true, y_pred)            
    
    
    def train_model(self,dataset, model_name='CNN_behav_cloning',valr=0.3,testr=0.1,shuff=True,col=None,opr=None,val=None,channels=None, save_path='./models/model.h5'):
        """
        Set training parameters. Does the learning process and saves the trained model.
        ----------
        Args:
            dataset (obj): The dataset of the preprocessed signal
            model_name (str, optional): Name of the deep learning model, which will be used. Defaults to 'CNN_behav_cloning'.
            valr (float, optional): Ratio of validation dataset size compared to the size if the whole dataset. Defaults to 0.3.
            testr (float, optional): Ratio of test dataset size compared to the size if the whole dataset.. Defaults to 0.1.
            shuff (bool, optional): If it's True the dataset will be shuffled before training. Defaults to True.
            col (str, optional): name of the column which needed to be filtered. Defaults to None.
            opr (str, optional): Given column's datapoints rel. Defaults to None.
            val (int): the value that describes the right data(eg.: data point should be > than val).Defaults to None.
            save_path(str): If self.is_save=True the trained model will be saved to this path.
        """
        
        #give the signal type to the generator. Options: ['cont' or 'ft_extr']
        signal_types = {'CNN_behav_cloning':'ft_extr', 'CNN_transf':'ft_extr', 'CNN_BiLSTM':'cont'}
        if signal_types.get(model_name)==None: raise ValueError('Model name is not valid!')
        
        #Set the choosed model type and get the right input shape
        if model_name=='CNN_transf':
            self.input_shape = dataset.ft_extracted.shape[1:]        
            model = CNN_transferlearn_model().assemble_full_model(self.input_shape)
        elif model_name=='CNN_behav_cloning':
            self.input_shape = dataset.ft_extracted.shape[1:]        
            model = CNN_behav_cloning_model().assemble_full_model(self.input_shape)
        elif model_name=='CNN_BiLSTM':
            self.input_shape = dataset.processed_data.shape[1:]   
            model = CNN_BiLSTM_model().assemble_full_model(self.input_shape, nlstm_units=20)
        else:
            raise ValueError('Unknown model_name at train_model. Change it to one of these:{0}'.format(signal_types.keys()))

        #Get train,validation and test indexes
        train_idx, valid_idx, test_idx = self.split_dataset(dataset,
                                                            valid_ratio=valr,
                                                            test_ratio=testr,
                                                            shuffle=shuff,
                                                            column=col,
                                                            operator=opr,
                                                            val=val)
        print(print_terminal(type='done',message='Dataset indexes has been generated.'))
        
        #Set up generators
        train_gen = self.generate_data(dataset, train_idx, is_training=True,channels=channels, stype = signal_types[model_name], batch_size=self.batch_size)
        valid_gen = self.generate_data(dataset, valid_idx, is_training=True,channels=channels, stype = signal_types[model_name], batch_size=self.valid_batch_size)
        print(print_terminal(type='done',message='Generators indexes has been initialized.'))
        
        #Get steps per epoch and validation steps parameters
        st_per_epoch = len(train_idx)//self.batch_size
        val_st       = len(test_idx)//self.valid_batch_size
        
        #Set up learning rate decay function
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.init_lr,
                                                                     decay_steps=self.decay_steps*st_per_epoch, 
                                                                     decay_rate=0.95,
                                                                     staircase=True)

        #Set up learning optimizer function
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, amsgrad=True) #added amsgrad=True
        #opt = tf.keras.optimizers.Adam(learning_rate=self.init_lr)
        
        model.compile(  optimizer=opt, 
                        loss={
                            'speed_output': self.mae, 
                            'head_dir_output': self.cyclical_mae_rad,
                            'position_X_output': self.euclidean_loss,
                            'position_Y_output': self.euclidean_loss
                            },
                        loss_weights={
                            'speed_output': 1.5, 
                            'head_dir_output': 1.5,
                            'position_X_output': 1.5,
                            'position_Y_output': 1.5
                            },
                        metrics={
                            'speed_output': self.mae,
                            'head_dir_output': self.cyclical_mae_rad,
                            'position_X_output': [self.euclidean_loss,'accuracy'],
                            'position_Y_output': [self.euclidean_loss,'accuracy'],                          
                    }
                        )
        print(print_terminal(type='done',message='Model compiling done.'))

        if st_per_epoch==0:
            print(print_terminal(type='error',message='Steps_per_epoch is zero. Lower the batch_size or feed more windows.'))
            raise ValueError("Steps_per_epoch is zero. Lower the batch_size or feed more windows.")
        if val_st==0:
            print(print_terminal(type='error',message='Validation_steps is zero. Lower the valid_batch_size or increase split ratio.'))
            raise ValueError("Validation_steps is zero. Lower the valid_batch_size or increase split ratio.")
        
        tf.saved_model.SaveOptions(experimental_custom_gradients=False)

        #Create callbacks array
        callbacks = []
        earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", restore_best_weights=True, patience=3)
        callbacks.append(earlystop)
        tensorboard_cp = TensorBoard(log_dir='{0}models/'.format(save_path[0]))
        callbacks.append(tensorboard_cp)
        model_cp = ModelCheckpoint(filepath='{0}models/'.format(save_path[0]), save_best_only=True, save_weights_only=True)
        callbacks.append(model_cp)
        print(print_terminal(type='done',message='Model callbacks have been created.'))                                               
        
        #Training
        print(print_terminal(type='run',message='Model starts learning...'))
        history = model.fit(train_gen,
                            steps_per_epoch=st_per_epoch,
                            epochs=self.epochs,
                            callbacks=callbacks,
                            shuffle=True,
                            validation_data=valid_gen,
                            validation_steps=val_st)
        print(print_terminal(type='done',message='Model learning done.'))

        #Save the best model
        #TODO: history1=np.load('history.npy',allow_pickle='TRUE').item()
        if self.is_save:
            save_path_model='{0}/models/{1}_{2}_model.h5'.format(save_path[0],save_path[1],model_name)
            rw.save_model(path=save_path_model,model=model)
            save_path_history='{0}/models/{1}_{2}_history.npy'.format(save_path[0],save_path[1],model_name)
            np.save(save_path_history,history.history)
            print(print_terminal(type='done',message='Model and history saved to: {}'.format(save_path)))
        
        return signal_types[model_name], history, model, train_idx, valid_idx, test_idx
        
        
class Evaluating:
    """
    Evaluating trained model and calculate parameters.
    """
    
    def __init__(self, window_size, is_save=True):
        self.window_size = window_size
        self.is_save     = is_save
    
    def load_model_from_h5(self, path=None):
        """
        Load trained model from .h5 file.
        ----------
        Args:
            path (str, optional): File path of trained model. Defaults to None.
        Returns:
            model(tensorflow model): loaded model.
        """
        print(print_terminal(type='done',message='Trained model loaded for evaulation.'))
        return tf.keras.models.load_model(path)
    
    @DataFile.path_check('w',required_format='.csv')
    def save_test_to_csv(self,path='./results/test_resuts.csv',df=None):
        """
        Save test results to csv.
        ----------
        Args:
            path (str, optional): Path of csv. Defaults to './results/test_resuts.csv'.
            df (pandas dataframe, optional): The dataframe with the results of evaulating. Defaults to None.
        """
        df.to_csv(path, index=False)
        print(print_terminal(type='done',message='Evaulation results saved.'))
    
    def euclidean_distance(self,x1,y1,x2=None,y2=None):
        """
        Calculate Euclidean distance based on 1 or two coordinate pairs.
        ----------
        Args:
            x1 (float): X coordiate of the first position.
            y1 (float): Y coordiate of the first position.
            x2 (float, optional): X coordiate of the second position.. Defaults to None.
            y2 (float, optional): Y coordiate of the second position.. Defaults to None.

        Returns:
            (float): calculated Euclidean distance
        """
        if x2!=None and y2!=None:
            point1 = np.array((x1, y1))
            point2 = np.array((x2, y2))
        else:
            point1 = np.array((x1))
            point2 = np.array((y1))
        sum_sq = np.sum(np.square(point1 - point2))
        return np.sqrt(sum_sq)
    
    def r2_score_pos(self, true_x, true_y, pred_x, pred_y):
        """
        Calculate the R2 score of the predicted and ground truth positions
        ----------
        Args:
            true_x (float): ground truth X coordinate
            true_y (float): ground truth Y coordinate
            pred_x (float): predicted X coordinate
            pred_y (float): pedicted Y coordinate

        Returns:
            (float): The R2 value
        """
        true = (true_x, true_y)
        pred = (pred_x, pred_y)
        return r2_score(true, pred)
    
    def cumulative_distribution(self,errors):
        """
        Calculate the cumulative distribution of errors.
        ----------
        Source:
        https://www.geeksforgeeks.org/how-to-calculate-and-plot-a-cumulative-distribution-function-with-matplotlib-in-python/
        ----------
        Args:
            errors (array): The errors in time
        Returns:
            cumulative,base,pdf (arrays): cummulative distributions, base and pdf
            
        """
        values, base = np.histogram(errors, bins=40)
        # finding the PDF of the histogram using count values
        pdf = values / sum(values)
        cumulative = np.cumsum(pdf)
        return cumulative,base,pdf
    
    
    def predict(self, model, test_generator, max_values):
        """
        It predicts a set of values and calculate all of the requied parameters.
        ----------
        Args:
            model (tensorflow model): The trained model
            valid_generator (generator): The validation generator, initialized earlier.
            valid_idx (array): indexes of the set of windows whitch will be used for evaulating the model.
        Returns:
            (pandas dataframe): All the calculated value in a dataframe.
        """
        
        count = 0
        #create dataframe with columns 
        df = pd.DataFrame(columns = ["time",
                                    "pred_speed", 
                                    "pred_head_dir", 
                                    "pred_pos_x", 
                                    "pred_pos_y", 
                                    "gtruth_speed", 
                                    "gtruth_head_dir", 
                                    "gtruth_pos_x", 
                                    "gtruth_pos_y",
                                    "speed_error",
                                    "speed_eucl_error",
                                    "head_dir_error",
                                    "head_dir_eucl_error",
                                    "pos_error_x",
                                    "pos_error_y",
                                    "pos_eucl_error",
                                    "r2_score_pos"
                                    ])
        
        for inp in test_generator:
            speed, head_dir, pos_x, pos_y = model.predict(inp[0])
            pred_speed = speed[0][0]*max_values[0]
            pred_head_dir = head_dir[0][0]*max_values[1]
            pred_pos_x = pos_x[0][0]*max_values[2]
            pred_pos_y = pos_y[0][0]*max_values[3]
            
            truth_speed = inp[1][0][0]*max_values[0]
            truth_head_dir = inp[1][1][0]*max_values[1]
            truth_pos_x = inp[1][2][0]*max_values[2]
            truth_pos_y = inp[1][3][0]*max_values[3]
            
            print('----> speed pred: {0}, truth speed: {1}'.format(pred_speed,truth_speed))
            
            
            df = df.append({'time': 0,#(test_idx[count]*self.window_size)+self.window_size,
                            'pred_speed' : pred_speed,
                            'pred_head_dir' : pred_head_dir, 
                            'pred_pos_x' : pred_pos_x,
                            'pred_pos_y' : pred_pos_y,
                            'gtruth_speed': truth_speed,
                            'gtruth_head_dir': truth_head_dir,
                            'gtruth_pos_x': truth_pos_x,
                            'gtruth_pos_y': truth_pos_y,
                            "speed_error": np.abs(pred_speed-truth_speed),
                            "speed_eucl_error": self.euclidean_distance(pred_speed,truth_speed),
                            "head_dir_error": np.abs(pred_head_dir-truth_head_dir),
                            "head_dir_eucl_error": self.euclidean_distance(pred_head_dir,truth_head_dir),
                            "pos_error_x": np.abs(pred_pos_x-truth_pos_x),
                            "pos_error_y": np.abs(pred_pos_y-truth_pos_y),
                            'pos_eucl_error':self.euclidean_distance(pred_pos_x,pred_pos_y,truth_pos_x,truth_pos_y),
                            'r2_score_pos': self.r2_score_pos(pred_pos_x,pred_pos_y,truth_pos_x,truth_pos_y)}, 
                            ignore_index = True)
            
            describe = df.describe()
            count+=1
        
        return df,describe
    
    
    def eval_pipeline(self,train_obj, test_idx, dataset, sign_type, nelements=100,inp_model=None, model_path=None,channels=None, save_path='./results/test_resuts.csv'):
        """
        Controll all the evaulation steps.
        ----------
        Args:
            model_path (str): path of trained model
            train_obj (_type_): Train() class object
            test_idx (_type_): index array of test windows
            dataset (obj): The dataset of the preprocessed signal.
            sign_type (_type_): type of signal which used for training
            nelements (int, optional): Number of windows from test dataset to evaulate. Defaults to 100.
            res_path (str, optional): File path where the calculated values will be saved as a csv. Defaults to './results/test_resuts.csv'.
        """
        print(print_terminal(type='run',message='Evaulation is running.'))
        #1. Load model
        if inp_model==None and model_path != None:
            model = self.load_model_from_h5(self, path=model_path)
        elif inp_model!=None:
            model = inp_model
        else:
            print(print_terminal(type='error',message='Missing model or model file path at evaulation.'))
        
        #2. Generate data
        test_gen = train_obj.generate_data(dataset, test_idx, is_training=False,channels=channels, stype=sign_type, batch_size=1)
        
        #3. Predicting
        pred_df,describe = self.predict(model, test_gen, max_values=train_obj.max_values)
        print(print_terminal(type='done',message='Evaulation done.'))
        
        #4. Save df
        if self.is_save:
            save_path_pred='{0}results/{1}_pred_results.csv'.format(save_path[0],save_path[1])
            self.save_test_to_csv(path=save_path_pred,df=pred_df)
            save_path_disc='{0}results/{1}_desc_results.csv'.format(save_path[0],save_path[1])
            self.save_test_to_csv(path=save_path_disc,df=describe)
            print(print_terminal(type='done',message='Evaulation saved.'))
        

def train_pipeline(dataset, mname, window_size, init_lr=1e-2, epochs=2, batch_size=10,
                   val_batch_size=10, decay_steps= 10, live_stream=False, valr=0.3,
                   testr=0.1, shuff=True, col=None, opr=None, val=None, channels=None,
                   chunk_size=None, is_gpu=None, is_paral=None, cutoff=None, tasks=None,
                   is_save=True, use_saver=True,save_path='./'):
    """
    It controls the training process.
    ----------
    Args:
        dataset (obj): The dataset of the preprocessed signal.
        mname (str): Name of the deep learning model
        window_size (int): Size of one model.
        init_lr (int, optional): Learning rate. Defaults to 1e-2.
        epochs (int, optional): Number of epochs. Defaults to 2.
        batch_size (int, optional): Training batch size. Defaults to 10.
        val_batch_size (int, optional): Validation batch size. Defaults to 10.
        decay_steps (int, optional): Decay ratio per steps. Defaults to 10.
        live_stream (bool, optional): If it's True the preprocessing will happens on the fly. Defaults to False.
        valr (float, optional): Validation ration. Defaults to 0.3.
        testr (float, optional): Test ratio. Defaults to 0.1.
        shuff (bool, optional): If it's True, the dataset will be shuffled. Defaults to True.
        col (str, optional): name of the column which needed to be filtered. Defaults to None.
        opr (str, optional): Given column's datapoints rel. Defaults to None.
        val (int): the value that describes the right data(eg.: data point should be > than val).Defaults to None.
        is_save (bool): if it's True the result (e.g.: trained model), will be saved.
        use_saver(bool): if it's True the progam will find the chunk size with the minimum data surplus during preprocess.
    """
    #Modify save_path for evaulate and model results
    folder= '/'.join(save_path.split('/')[:-1])+'/'
    format= pathlib.Path(save_path).suffix
    if len(format)==0:
        fname= str(save_path.split('/')[-1])
    else:
        fname= str(save_path.split('/')[-1][:-len(format)])
    save_path = [folder,fname,format]
    
    #Initialize Training class
    train = Training(init_lr=init_lr, 
                    epochs=epochs, 
                    batch_size=batch_size,
                    valid_batch_size=val_batch_size,
                    dataset=dataset,
                    decay_steps= decay_steps,
                    live_stream=live_stream,
                    window_size=window_size,
                    chunk_size=chunk_size,
                    is_gpu=is_gpu,
                    is_paral=is_paral,
                    cutoff=cutoff,
                    tasks=tasks,
                    is_save=is_save,
                    use_saver=use_saver)
    
    
    print(print_terminal(type='done',message='Training class initialized.'))
    
    (sign_type, 
     history, 
     model, 
     train_idx, 
     valid_idx,
     test_idx) = train.train_model(dataset, 
                                    model_name=mname,
                                    valr=valr,
                                    testr=testr,
                                    shuff=shuff,
                                    col=col,
                                    opr=opr,
                                    val=val,
                                    channels=channels,
                                    save_path=save_path)
    
    eval = Evaluating(window_size=window_size, is_save=is_save)
    eval.eval_pipeline(inp_model=model,
                       train_obj=train,
                       test_idx=test_idx,
                       dataset=dataset,
                       sign_type=sign_type,
                       nelements=2,
                       channels=channels,
                       save_path=save_path)
    
    return history, train_idx, valid_idx, test_idx
