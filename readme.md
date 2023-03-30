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

| Variable | Description | Possible Values |
| --- | --- | --- |
| base_path | Gives the source folder where the sources were saved | Any valid folder path |
| is_test | If it's True no output data will be saved and make a minimum runtime to check functions | True or False |
| is_save | If it's True all results will be saved (eg.: plots, models, etc.) | True or False |
| source_file_path | Gives the source path (base_path + 'source file name') | Any valid file path |
| original_srate | Sampling rate of the raw signal in Hz | 30000 |
| led_srate | Sampling rate of the raw LED signal in Hz | 1000 |
| new_srate | Downscaled target sampling rate in Hz | 10000 |
| window_size | Size of one window in milliseconds | 2000 |
| chunk_size | Size of one chunk in milliseconds | 30000 |
| is_save_images | If it is True the plots will be saved | True or False |
| image_save_dir | The name of the target folder where the plots will be saved | Any valid folder path |
| start_point | It gives the start point of the plotted signal parts in ms | Any positive integer |
| end_point | It gives the end point of the plotted signal parts in ms | Any positive integer greater than start_point |
| tasks['Cheby_band'] | Chebyshev bandpass filter parameters: [lowcut, highcut, order] | Normally [2,250,5] |
| tasks['narrow_filt'] | Notch filter parameters: [notch frequency, quality factor] | Normally [50,20] |
| tasks['down_samp'] | Downsample parameter: [new sampling rate] | [new_srate] |
| tasks['detrend'] | Detrend parameter: True or False | True or False |
| tasks['roll_mean'] | Rolling mean parameter: [windows factor] | Normally [20] |
