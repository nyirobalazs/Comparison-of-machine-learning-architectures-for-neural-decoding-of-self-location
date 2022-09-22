import os
import h5py
import numpy as np
import textwrap
import json
import pathlib
from datetime import date,datetime
from kaleido.scopes.plotly import PlotlyScope
from functools import wraps


def print_terminal(type,message, answer=None):
    """
    Print messages to the terminal.
    ----------
    Args:
        type (str): type of message.Options: ['error','inp','done','run']
        message (str): The actual message.
    Returns:
        output(str): output message which will be printed.
    """
    
    COMMANDS = {
    'error': (31, '[!] ', 'Error: '),
    'inp': (34, '[?] ', 'Question: '),
    'done': (32, '[🗸] ', ''),
    'run': (33, '[>>]', '')    
    }
    
    command = COMMANDS[type]
    current_time = datetime.now().strftime("%H:%M:%S")
    if type=='inp' and answer!=None:
        message = '{0} | Answer: {1}'.format(message,answer)
    
    output = '\033[{0};1m{1}  \033[0m|{2}|  \033[{0};1m {3}\033[0;{0}m{4}\033[0m'.format(command[0],command[1],current_time,command[2],message)
        
    return output

print(print_terminal(type='done',message='Python package initialization done in read_write.'))

class Data:
    
    def __init__(self):
        
            self.raw_signal              = []
            self.ft_extracted            = [] #feature extracted
            self.processed_data          = [] #signal before feature extraction
            self.head_dir                = []
            self.speed                   = []
            self.position                = []
            self.raw_led_positions       = []
            self.tetrode_timestamps      = []
            self.processed_timestamps    = []
            self.led_timestamps          = []
            self.preprocessed            = False
            self.tetrode_srate           = int()
            self.led_srate               = int()
            self.preprocess_steps_values = {}
            self.good_channels           = []
            self.bad_channels            = []
            
    def get_info(self):
        """
        Get settings from dataset.
        ----------
        Returns:
            info(dict): concatenated infos 
        """

        info = {'preprocessed' : self.preprocessed,
                'tetrode_srate': self.tetrode_srate,
                'led_srate': self.led_srate,
                'preprocess_steps_values': self.preprocess_steps_values,
                'good_channels': self.good_channels,
                'bad_channels': self.bad_channels            
                }
        return info
        
        

class DataFile:
    """
    Basic dataset class of the whole program.
    """
    
    def __init__(self, 
                 original_srate=None,
                 led_srate=None 
                 ):
        
        self.source_path = ''
        self.output_path = ''
        self.original_srate = original_srate
        self.led_srate = led_srate
        self.processed = False
    
    
    def is_exist(self, path):
        """
        Checks if the file at the given past exist
        ----------
        Args:
            path (str): file's path

        Returns:
            bool: True if exist and False if it's not
        """
        
        return os.path.exists(path)
    
    
    def path_check(mode,required_format=None):
        """
        A decorator which checks the validity of the given file path,
        the given file type and helps to change if it's necessary.
        ----------
        Args:
            mode (str): 'r' = read | 'w'=write
            required_format (str, optional): The possible file extension options. Defaults to None.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(self=None,path=None,*args,**kwargs):
                while True:
                    #Separating from file path the folder, format and file names
                    folder= '/'.join(path.split('/')[:-1])+'/'
                    format= pathlib.Path(path).suffix
                    if len(format)==0:
                        fname= str(path.split('/')[-1])
                    else:
                        fname= str(path.split('/')[-1][:-len(format)])
                    
                    #If there is a required format type check the given suffix is correct.
                    if (required_format!=None):
                        #In case of only one option
                        if type(required_format)==str:
                            if format!=required_format:
                                path = input('Please give a file path with a {0} suffix.'.format(required_format)) 
                                print(print_terminal(type='inp',message='Give new file path with {0}'.format(required_format),answer=path))
                        #if there are multiple option
                        elif type(required_format)==list or type(required_format)==tuple:
                            if not (format in list(required_format)):
                                path = input('Please give a file path with one of these suffixes: {0}'.format(', '.join(map(str, list(required_format))))) 
                                print(print_terminal(type='inp',message='Give new file path with {0}'.format(', '.join(map(str, list(required_format)))),answer=path))
                    #If folder not exist, create the right folder(s)
                    if not os.path.isdir(folder):
                        os.makedirs(folder)
                        pass
                    elif not os.path.isfile(path): #if given path NOT leads to an existing file
                        if mode.lower()=='r': #if mode=read --> we need a file so ask for a new path
                            path = input('File not exists.Select an existing one.')
                            print(print_terminal(type='inp',message='Give existing path.',answer=path))
                            pass
                        elif mode.lower()=='w': #if mode=write
                            if len(format)!=0 and len(fname)!=0: #if everything right--> return
                                break
                            else: #if no file format or file name is given --> as for a new filepath
                                path = input('Missing filename or format. Give a correct path.')
                                print(print_terminal(type='inp',message='Give correct path.',answer=path))
                                pass
                    elif mode.lower()=='r': #if given path LEADS to an existing file and mode=read
                        if len(format)!=0 and len(fname)!=0: #if mode=read -->return
                            break
                    elif mode.lower()=='w': #if mode==write --> add current time to the filename -->return
                        tdate = date.today()
                        time = datetime.now()
                        unique_ext = '{0}_{1}'.format(tdate.strftime("%d_%m_%Y"),time.strftime("%H_%M"))
                        path = '{0}{1}_{2}{3}'.format(folder,fname,unique_ext,format)
                        print(print_terminal(type='run',message='File path changed to:{}'.format(path)))
                        break
                if self==None:
                    return func(path=path,*args,**kwargs)
                else:
                    return func(self,path=path,*args,**kwargs)
            return wrapper
        return decorator
    
    
    def create_or_update(self, hdf5_file, dataset_name, dataset_shape, dataset_type, dataset_value):
            """
            Create or update dataset in HDF5 file.
            ----------
            DeepInsight Toolbox
            © Markus Frey
            https://github.com/CYHSM/DeepInsight
            Licensed under MIT License
            
            Parameters
            ----------
            hdf5_file : File
                File identifier
            dataset_name : str
                Name of new dataset
            dataset_shape : array_like
                Shape of new dataset
            dataset_type : type
                Type of dataset (np.float16, np.float32, 'S', etc...)
            dataset_value : array_like
                Data to store in HDF5 file
            """
            max_shape_tuple = tuple([None]*len(dataset_shape))
            if not dataset_name in hdf5_file:
                hdf5_file.create_dataset(dataset_name, dataset_shape, dtype= dataset_type, compression="gzip", compression_opts=4, chunks=True, maxshape=max_shape_tuple)
                hdf5_file[dataset_name][:] = dataset_value
            else:
                if hdf5_file[dataset_name].shape != dataset_shape:  
                    new_shape = tuple(hdf5_file[dataset_name].shape[i] if hdf5_file[dataset_name].shape[i]==dataset_shape[i] else hdf5_file[dataset_name].shape[i]+dataset_shape[i] for i in range(len(hdf5_file[dataset_name].shape)))
                    hdf5_file[dataset_name].resize(new_shape)
                    hdf5_file[dataset_name][-dataset_value.shape[0]:] = dataset_value
            hdf5_file.flush()
            
    @path_check('w',required_format='.h5')
    def write_raw_h5_file(self, path='./', dataset=None):
        """
        Write raw data's variables into a raw .h5 file.
        ----------
        Args:
            path (str): gives input file's path
            dataset (obj,optional): dataset with variables. Default is None.
        """
        self.output_path = path
        
        print(print_terminal(type='run',message="Raw .h5 file writing has been started"))
        
        h5_file = h5py.File(path, mode='a')
        # Get size of wavelets
        output_channels = dataset.raw_signal.shape[1]
        # write given dataelements into the file
        self.create_or_update(h5_file, dataset_name="outputs/raw_tetr_sign", 
                              dataset_shape=[dataset.raw_signal.shape[0], output_channels], dataset_type=np.float64, dataset_value=dataset.raw_signal)        
        self.create_or_update(h5_file, dataset_name="outputs/raw_led_positions",
                            dataset_shape=[dataset.raw_led_positions.shape[0], dataset.raw_led_positions.shape[1]], dataset_type=np.float64, dataset_value=dataset.raw_led_positions)
        self.create_or_update(h5_file, dataset_name="outputs/tetr_timestamps",
                            dataset_shape=[dataset.tetrode_timestamps.shape[0], ], dataset_type=np.float64, dataset_value=dataset.tetrode_timestamps)
        self.create_or_update(h5_file, dataset_name="outputs/led_timestamps",
                            dataset_shape=[dataset.led_timestamps.shape[0], ], dataset_type=np.float64, dataset_value=dataset.led_timestamps)
        self.create_or_update(h5_file, dataset_name="outputs/preprocessed",
                            dataset_shape=[1,], dataset_type=np.bool8, dataset_value=dataset.preprocessed)
        self.create_or_update(h5_file, dataset_name="outputs/good_channels",
                            dataset_shape=[dataset.good_channels.shape[0],], dataset_type=np.int16, dataset_value=dataset.good_channels)
        self.create_or_update(h5_file, dataset_name="outputs/bad_channels",
                            dataset_shape=[dataset.bad_channels.shape[0],], dataset_type=np.int16, dataset_value=dataset.bad_channels)
        self.create_or_update(h5_file, dataset_name="outputs/raw_tetr_sampling_rate",
                            dataset_shape=[1,], dataset_type=np.float16, dataset_value=dataset.tetrode_srate)
        self.create_or_update(h5_file, dataset_name="outputs/led_sampling_rate",
                            dataset_shape=[1,], dataset_type=np.float16, dataset_value=dataset.led_srate)
        
        h5_file.flush()
        h5_file.close()
        print(print_terminal(type='done',message='Raw dataset writing to .h5 file is done.'))
        
    @path_check('w',required_format='.h5')    
    def write_processed_h5_file(self, path='./', dataset=None):
        """
        Write processed data's variables into a .h5 file.
        ----------
        Args:
            path (str): gives input file's path
            dataset (obj,optional): dataset with variables.Default is None.
        """
        self.output_path = path
        
        print(print_terminal(type='run',message="Processed .h5 file writing has been started"))
        
        h5_file = h5py.File(path, mode='a')
        # write given dataelements into the file
        if len(dataset.ft_extracted.shape)==4:
            self.create_or_update(h5_file, dataset_name="outputs/ft_extracted", 
                                dataset_shape=dataset.ft_extracted.shape, dataset_type=np.float64, dataset_value=dataset.ft_extracted)
        else:
            self.create_or_update(h5_file, dataset_name="outputs/feature_extracter_data", dataset_shape=[1,], dataset_type=np.float64, dataset_value=None)  
        self.create_or_update(h5_file, dataset_name="outputs/processed_data", 
                              dataset_shape=dataset.processed_data.shape, dataset_type=np.float64, dataset_value=dataset.processed_data)
        self.create_or_update(h5_file, dataset_name="outputs/raw_led_positions",
                            dataset_shape=[dataset.raw_led_positions.shape[0], dataset.raw_led_positions.shape[1]], dataset_type=np.float64, dataset_value=dataset.raw_led_positions)
        self.create_or_update(h5_file, dataset_name="outputs/led_timestamps",
                            dataset_shape=[dataset.led_timestamps.shape[0], ], dataset_type=np.float64, dataset_value=dataset.led_timestamps)
        self.create_or_update(h5_file, dataset_name="outputs/processed_timestamps",
                            dataset_shape=dataset.processed_timestamps.shape, dataset_type=np.float64, dataset_value=dataset.processed_timestamps)
        self.create_or_update(h5_file, dataset_name="outputs/preprocessed",
                            dataset_shape=[1,], dataset_type=np.bool8, dataset_value=dataset.preprocessed)
        self.create_or_update(h5_file, dataset_name="outputs/tetrode_srate",
                            dataset_shape=[1,], dataset_type=np.int16, dataset_value=dataset.tetrode_srate)
        self.create_or_update(h5_file, dataset_name="outputs/good_channels",
                            dataset_shape=dataset.good_channels.shape, dataset_type=np.int16, dataset_value=dataset.good_channels)              
        
        json_path = '{0}_tasks{1}'.format(path[:-3],'.json')
        with open(json_path, "w") as outfile:
            json.dump(dataset.preprocess_steps_values, outfile)
        
        h5_file.flush()
        h5_file.close()
        print(print_terminal(type='done',message='Raw dataset writing to .h5 file is done.'))

    
    @path_check('r',required_format='.nwb')
    def load_nwb_file(self, path='./'):
        """
        Loads the raw data from the nwb file
        ----------
        Return:
            ds: data object with the raw datas
        """
        self.source_path = path
        
        if self.is_exist(self.source_path):
            
            # Create data object and open the file
            ds = Data()
            raw_data              = h5py.File(self.source_path, mode='r')
            #Load each variable from the file            
            record_key            = list(raw_data['acquisition']['timeseries'].keys())[0]
            process_key           = list(raw_data['acquisition']['timeseries'][record_key]['continuous'].keys())[0]
            ds.raw_signal         = raw_data['acquisition']['timeseries'][record_key]['continuous'][process_key]['data']
            ds.tetrode_timestamps = raw_data['acquisition']['timeseries'][record_key]['continuous'][process_key]['timestamps']
            led_positions         = raw_data['acquisition']['timeseries'][record_key]['tracking']['ProcessedPos']
            settings              = raw_data['general']['data_collection']['Settings']
            ds.led_timestamps     = led_positions[:, 0]
            ds.raw_led_positions  = led_positions[:, 1:5]
            bad_channels          = settings['General']['badChan']
            ds.bad_channels       = [int(n) for n in bad_channels[()].decode('UTF-8').split(',')]
            ds.good_channels      = np.delete(np.arange(0, 128), ds.bad_channels)
            ds.tetrode_srate      = self.original_srate
            ds.led_srate          = self.led_srate
            print(print_terminal(type='done',message='The signal loading from raw .nwb file is done.'))
                        
        else:
            raise ValueError(print_terminal(type='error', message='.nwb file is not exist.'))
        
        return ds
    
    @path_check('r',required_format='.h5')
    def load_raw_h5(self, path='./'):
        """
        Load raw data from .h5 file.
        ----------
        Args:
            path (str): path of the raw file
        Returns:
            ds (obj): dataset
        """
                
        self.source_path = path
        
        if self.is_exist(self.source_path):
            
            # Create dataset and open .h5 file
            ds = Data()
            raw_data = h5py.File(self.source_path, mode='r')
            # Read variables
            ds.raw_signal = raw_data["outputs/raw_tetr_sign"]
            ds.raw_led_positions = raw_data["outputs/raw_led_positions"]
            ds.tetrode_timestamps = raw_data["outputs/tetr_timestamps"]
            ds.led_timestamps = raw_data["outputs/led_timestamps"]
            ds.preprocessed = raw_data["outputs/preprocessed"]
            ds.good_channels = raw_data["outputs/good_channels"]
            ds.bad_channels = raw_data["outputs/bad_channels"]
            ds.tetrode_srate = raw_data["outputs/raw_tetr_sampling_rate"]
            ds.led_srate = raw_data["outputs/led_sampling_rate"]
            print(print_terminal(type='done',message='File loading from raw .h5 file is done.'))
        
        elif not self.is_exist(self.source_path):
            raise ValueError(print_terminal(type='error',message=' Raw .h5 file is not exist'))
           
        return ds
    
    @path_check('r',required_format='.h5')
    def load_processed_h5(self, path='./'):
        """
        Load processed data from .h5 file.
        ----------
        Args:
            source_path (str): path of the raw file
        Returns:
            ds (obj): dataset
        """        
        
        self.source_path = path
        
        if self.is_exist(self.source_path):
            
            # Create dataset and open .h5 file
            ds = Data()
            raw_data = h5py.File(self.source_path, mode='r')
            # Read variables
            ds.ft_extracted = raw_data["outputs/ft_extracted"]
            ds.processed_data = raw_data["outputs/processed_data"]
            ds.raw_led_positions = raw_data["outputs/raw_led_positions"]
            ds.led_timestamps = raw_data["outputs/led_timestamps"]
            ds.processed_timestamps = raw_data["outputs/processed_timestamps"]
            ds.preprocessed = raw_data["outputs/preprocessed"]
            ds.good_channels = raw_data["outputs/good_channels"]
            ds.tetrode_srate = raw_data["outputs/tetrode_srate"]
            print(print_terminal(type='done',message='File loading from processed .h5 file is done.'))
        
        elif not self.is_exist(self.source_path):
            raise ValueError(print_terminal(type='error',message='Proecessed .h5 file is not exist'))
        
        return ds
    
@DataFile.path_check('w', required_format='.h5')    
def save_model(path='./models/model.h5', model=None):
    """
    Save trained model into .h5 and .json files.
    ----------
    Args:
        path (str, optional): Path of output files. Defaults to './model'.
        model (Model class, optional): trained model. Default is None.
    """
    #separate filepath from file format.
    file_extension = pathlib.Path(path).suffix
    path = path[:-len(file_extension)]

    #save with tensorflow's saving function in .h5 and .json format too.
    model.save_weights(path+'.h5', True)
    with open(path+'.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
    

    
@DataFile.path_check('w',required_format=['.jpg','.jpeg','.svg','.png']) 
def save_plot(path='./',fig=None):
    """
    Save plots.
    ----------
    Args:
        path (str, optional): filepath. Defaults to './'.
        fig (_type_, optional): plt figure option. Defaults to None.
    """
    
    #get file format to set the writing function's format parameter.
    file_format = pathlib.Path(path).suffix
    #writing with PlotlyScope function. Other functions lead to errors
    scope = PlotlyScope()      
    with open(path, "wb") as f:
        f.write(scope.transform(fig, format=file_format[1:]))
        
        
@DataFile.path_check('w',required_format=['.csv','.h5','.mat','.nwb'])    
def save_variables(path='./',
                   tasks=None, 
                   tetr_srate=None,
                   led_srate=None,
                   wind_size=None,
                   chunk_size=None,
                   cutoff_size=None,
                   mod_name=None,
                   lr_rate=None,
                   nepochs=None,
                   tbatch=None,
                   val_batch=None,
                   lr_ind=None,
                   val_ind=None,
                   ts_ind=None,
                   shuffl=None,
                   fparam=None,
                   frel=None,
                   flim=None,
                   is_save=True):
    """
    This function saves all the important variables and settings to make easier the later version tracking and analysis.
    """   
    
    variables = {
        'save_datetime' : datetime.now().strftime("%d/%m/%Y-%H:%M:%S"),
        'tetrode_srate' : tetr_srate,
        'led_srate'     : led_srate,
        'windows_size'  : wind_size,
        'chunk_size'    : chunk_size,
        'cutoff_size'   : cutoff_size,
        'model_name'    : mod_name,
        'learning_rate' : lr_rate,
        'epochs'        : nepochs,
        'train_batch'   : tbatch,
        'valid_batch'   : val_batch,
        'learn_indexes' : lr_ind,
        'valid_indexes' : val_ind,
        'test_indexes'  : ts_ind,
        'is_shuffle'    : shuffl,
        'filter_param'  : fparam,
        'filter_relation': frel,
        'filter_limit'  : flim,
        'preproc_tasks' : tasks}
    
    if is_save:
        #separate parts from path
        folder= '/'.join(path.split('/')[:-1])+'/results/'
        if not os.path.exists(folder): #if directory not exist create one
            os.makedirs(path)
        format= pathlib.Path(path).suffix
        if len(format)==0:
            fname= str(path.split('/')[-1])
        else:
            fname= str(path.split('/')[-1][:-len(format)])
        save_path = '{0}{1}.json'.format(folder,fname)#create new filename
        #save dict to json        
        with open(save_path, "w") as outfile:
            json.dump(variables, outfile)
        print(print_terminal(type='done',message='Variables and settings sucessfuly saved.'))
    else:
        print(print_terminal(type='error',message='Variables and settings were not saved.'))
    
    
    

@DataFile.path_check('r',required_format=['.nwb','.h5'])
def data_load_pipeline(path='./', original_srate=None, led_srate=None, is_test=False):
    """
    It manages the dataload and conversion process, with different filetypes.
    ----------
    Args:
        path (str): path of the raw file.
        original_srate (int): sampling rate of the raw tetrode data.
        led_srate (int): sampling rate of the raw led position data.
        is_test (bool): if it's True, no output will be saved.
        
    Returns:
            dataset (obj): dataset.
            path (str): optionally modificated source file path.
    """
    
    load_code = 0 #if =0 -->preprocessing needed | if =1 --> continue from learning
    #Check is sampling rates exist.
    if (original_srate==None) or (led_srate==None):
        raise TypeError(print_terminal(type='error',message='Missing original_srate or original_srate parameter(s).'))
    
    df = DataFile(original_srate, led_srate)
    folder= '/'.join(path.split('/')[:-1])+'/'
    format= pathlib.Path(path).suffix
    if len(format)==0:
        fname= str(path.split('/')[-1])
    else:
        fname= str(path.split('/')[-1][:-len(format)])

    #if file is not a preprocessed signal
    if not ('preprocessed' in fname.split("_")):
        
        print(print_terminal(type='run',message='The source file is a raw file.'))
        
        #if file is an .nwb file
        if path.endswith('.nwb'):
            
            #infos to the user
            print(textwrap.dedent("""
                The file has the extension .nwb. The program can generate a more
                compact .h5 raw signal format from the current file, that contains
                the necessary parameters to simplify later use.
                """))
            while True:
                if is_test:
                    answer='n'
                    print(print_terminal(type='run',message='Test mode is active. No file will be saved.'))
                else:
                    answer = input("Do  you want to convert the current .nwb file into .h5 file before processing?: (Y/N)").lower()
                    print(print_terminal(type='inp',message='Convert raw .nwb to .h5?', answer=answer))
                    
                if (answer == 'y') or (answer == 'yes'): #if the user wants to convert it
                    dataset = df.load_nwb_file(path=path)
                    load_code = 0
                    df.write_raw_h5_file(path='{0}{1}.h5'.format(folder,fname), dataset=dataset)
                    break
                elif (answer == 'n') or (answer == 'no'): #if the user do not want to convert the input file
                    dataset = df.load_nwb_file(path=path)
                    load_code = 0
                    break
                else:
                    print(print_terminal(type='error',message='Please answere with Y/Yes or N/no'))
                        
        #if the file is a .h5 file            
        elif path.endswith('.h5'):
                        
            processed_name = '{0}preprocessed_{1}{2}'.format(folder,fname,format)
            
            if os.path.exists(processed_name): #is there any preprocessed file in the root directory
                while True:
                    answer = input("There is a prerpocessed file with the same name. Do you want to use that one?: (Y/N)").lower()
                    print(print_terminal(type='inp',message='Existing processed file. Use that one?', answer=answer))
                    if (answer == 'y') or (answer == 'yes'):
                        #change file source to preprocessed file
                        df.processed = True
                        path = processed_name
                        break
                    elif (answer == 'n') or (answer == 'no'):
                        #just read the .h5 file
                        dataset = df.load_raw_h5(path=path)
                        load_code = 0
                        break
                    else:
                        print(print_terminal(type='error',message='Please answere with Y/Yes or N/no'))
            elif df.processed == False: 
                dataset = df.load_raw_h5(path=path)
                load_code = 0

    elif ('preprocessed' in fname.split("_")):
                   
        dataset = df.load_processed_h5(path=path)
        load_code = 1
    
    print(print_terminal(type='done',message='File loading has been finished'))

    return dataset, path, load_code

