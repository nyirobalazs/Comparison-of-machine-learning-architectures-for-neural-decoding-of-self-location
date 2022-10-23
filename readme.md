# **Comparison of machine learning architectures for neural decoding of self-location**
___

In this project, a set of tools has been developed to decode the position, head orientation and velocity of an animal from neurophysiological signals recorded from raw place cells. The program can use behaviour cloning, transfer learning or CNN-BiLSTM architecture for machine learning. 

My thesis can be read **[HERE](https://github.com/nyirobalazs/thesis/blob/main/Comparison_of_machine%20learning_Balazs_Agoston_Nyiro.pdf)**

![figure](./assets/neural%20decoding-.png)


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

