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


# Example project

In the example.ipynb file you can find an example project.

## How to run

To run the project, follow these steps:

1. Clone or download this repository to your local machine.
2. Install the required packages listed in the requirements.txt file.
3. Open the example.ipynb file and edit the variables and settings as needed.
4. Run the segments and wait for the results.

## Variables and Settings

The following variables and settings are used in the project:

| Variable | Description | Possible Values | Value Type |
| --- | --- | --- | --- |
| base_path | Gives the source folder where the sources were saved | Any valid folder path | String |
| is_test | If it's True no output data will be saved and make a minimum runtime to check functions | True or False | Boolean |
| is_save | If it's True all results will be saved (eg.: plots, models, etc.) | True or False | Boolean |
| source_file_path | Gives the source path (base_path + 'source file name') | Any valid file path | String |
| original_srate | Sampling rate of the raw signal in Hz | 30000 (based on your signal) | Integer |
| led_srate | Sampling rate of the raw LED signal in Hz | 1000 (based on your signal) | Integer |
| new_srate | Downscaled target sampling rate in Hz | 10000 (based on your signal) | Integer |
| window_size | Size of one window in milliseconds | 2000 (based on your signal) | Integer |
| chunk_size | Size of one chunk in milliseconds | 30000 (based on your signal) | Integer |
| is_save_images | If it is True the plots will be saved | True or False | Boolean |
| image_save_dir | The name of the target folder where the plots will be saved | Any valid folder path | String |
| start_point | It gives the start point of the plotted signal parts in ms | Any positive integer | Integer |
| end_point | It gives the end point of the plotted signal parts in ms | Any positive integer greater than start_point | Integer |
| tasks |  A dictionary containing the required preprocessing steps and their settings. The possible keys are 'Cheby_band', 'narrow_filt', 'down_samp', 'detrend', 'roll_mean', 'normalize', and 'feature_extract'. | If you want to switch off a preprocessing step, write False to the parameters section. See default settings below. | False or a list |
| <ul><li> 'Cheby_band' | Chebyshev bandpass filter parameters: [lowcut, highcut, order] | Normally [2,250,5] or any valid filter parameters within the range of the signal frequency spectrum. The order must be a positive integer. The lowcut and highcut must be positive floats. The lowcut must be lower than the highcut.  | List of floats and integer |
| <ul><li> 'narrow_filt' | Notch filter parameters: [notch frequency, quality factor] | Normally [50,20] or any valid filter parameters within the range of the signal frequency spectrum. The notch frequency must be a positive float. The quality factor must be a positive float greater than zero.  | List of floats |
| <ul><li> 'down_samp' | Downsample parameter: [new sampling rate] | [new_srate] or any valid sampling rate lower than the original sampling rate. The new sampling rate must be a positive integer.  | List of integer |
| <ul><li> 'detrend' | Detrend parameter: True or False. If it is True, a linear detrending will be applied to the signal.  | True or False  | Boolean |
| <ul><li> 'roll_mean' | Rolling mean parameter: [windows factor]. If it is a positive integer greater than zero, a rolling mean with a window size of windows factor times the window size will be applied to the signal. If it is zero, no rolling mean will be applied. | Normally [20].  Any positive integer or zero. | Integer |


## License

This project is licensed under the MIT License - see the LICENSE file for details.


