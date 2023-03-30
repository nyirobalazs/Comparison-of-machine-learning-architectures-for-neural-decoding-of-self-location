# **Comparison of machine learning architectures for neural decoding of self-location**
___

In my master dissertation project at the University of Nottingham, a set of tools has been developed to decode the position, head orientation and velocity of an animal from neurophysiological signals recorded from raw place cells. The program can use behaviour cloning, transfer learning or CNN-BiLSTM architecture for machine learning. 

My thesis can be read **[HERE](https://github.com/nyirobalazs/thesis/blob/main/Comparison_of_machine%20learning_Balazs_Agoston_Nyiro.pdf)**

![figure](./assets/neural%20decoding-_.png)


### **Data**

Raw neural recordings from CA1 in rodents using tetrode recordings from Markus Frey et. al.(2021) experiment: https://figshare.com/articles/dataset/Tetrode_recordings_from_CA1/14909766?backTo=/collections/DeepInsight_-_Data_sharing/5486703


### **Features**

- read raw signal from a .nwb or a .h5 files
- convert raw .nwb file into a raw .h5 format file in the required data system
- preprocess files with optional bandpass, line noise, detrend, rolling window mean filters and downsampling
- write preprocessed files into a .h5 file
- use single CPU, paralell computing(on multi-core CPU) or use GPU for preprocessing, based on the fastest calculated processing mode
- learn decoding from neural code by:
    - Behaviour cloning
    - Transfer learning (Efficientnet B0 with imagenet weights)
    - CNN-BiLSTM architecture
- use preprocessed data from ram or file for training or preprocess on-the-flight
- save best weights
- make predictions with the trained network
- evaulate model (save results)



Decoding Rat's Location in Space
This project aims to help neuroscience researchers decode rats' location in space based on tetrode data. The data is stored in an hdf5 file, and this project provides various pre-processing and deep learning models to decode the rat's position.

Variables
base_path: The base path where the source files are saved.
is_test: A boolean variable that determines if the output data should be saved or not. If it's True, no output data will be saved, and it will make a minimum runtime to check functions.
is_save: A boolean variable that determines if all results should be saved (e.g. plots, models, etc.).
Signal File
source_file_path: The path to the processed hdf5 file containing the raw data.
original_srate: The sampling rate of the raw signal in Hz, defined as 30000.
led_srate: The sampling rate of the raw LED signal in Hz, defined as 30000.
new_srate: The downscaled target sampling rate in Hz, defined as 10000.
window_size: The size of one window in milliseconds, defined as 2000.
chunk_size: The size of one chunk in milliseconds, defined as 30000.
Visualization
is_save_images: A boolean variable that determines if the plots will be saved.
image_save_dir: The name of the target folder where the plots will be saved, defined as './images/'.
start_point: The start point of the plotted signal parts in ms, defined as 0.
end_point: The end point of the plotted signal parts in ms, defined as 1000.
Preprocess
chunk_perc: If it's None, the whole dataset will be preprocessed. If the value is between 1 and 100, it will process a percentage of all chunk_ only.
channels: The channels which needed to be processed. If None, all good channels will be processed; tuple->(from,to); int->(0,to); list or np.array -> needed channels.
tasks: A dictionary that gives the required preprocessing steps and their settings.
Options:
'Cheby_band': [lowcut, highcut, order]. Normally [2,250,5].
'narrow_filt': [notch frequncy, quality factor]. Normally [50,20].
'down_samp': [new sampling rate]. Leave it [new_srate].
'detrend': True or False.
'roll_mean': [windows factor]. Normally [20].
'normalize': True or False.
'feature_extract': [lowcut, highcut, number of voices, scaletype ('log-piecewise' or 'linear' or log), wavelet('morlet'), rescaled size]. Normally [2,250,5,'log-piecewise','morlet',64].
If you want to switch off a preprocessing step, write False to the parameters section (e.g.: 'down_samp':False). If rescaled size=0, it won't resize the output matrix.
cutoff: The size of the extra part that will be cut off from the beginning and the end of the chunk to avoid edge effects in milliseconds.
use_gpu: If it is True, the GPU will be used for preprocessing if it's available.
use_paral: If it's True, parallel computing will be used to speed up preprocessing
