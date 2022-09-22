import os
import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample, butter, lfilter, iirnotch, filtfilt, detrend
from tqdm.notebook import trange,tqdm
import pathlib
import torch
torch.cuda.empty_cache()
import gc
from skimage.transform import resize
from ssqueezepy import cwt as ss_cwt, Wavelet
from ssqueezepy.utils import make_scales
from wavelets import Morlet
from wavelets import cwt as wt_cwt
from joblib import Parallel, delayed
import numba as nb

from read_write import DataFile, print_terminal
print(print_terminal(type='done',message='Python package initialization done in preprocess.'))   

class Preprocess():
    
    def __init__(self, 
                 dataset,
                 window_size = 1000,
                 chunk_size = 50000,
                 is_gpu     = False,
                 is_paral   = True,
                 cutoff     = 0,
                 ml_in_use  = False,
                 use_saver  = True):
        
        self.window_size             = window_size
        self.chunk_size              = chunk_size
        self.good_ch                 = list(dataset.good_channels)
        self.count_cpu               = os.cpu_count()
        self.saver                   = use_saver #If it's True the progam will find the chunk size with the minimum data surplus
        if torch.cuda.is_available(): #if there is an available GPU it will use that unless it was set to not use it
            self.gpu = True
        else:
            self.gpu = False
        self.set_cpu_gpu_mode(is_gpu, is_paral)
        
        """If there is a >0 cutoff size it will set an extra zone at the start and end of the chunks, 
        and cut them off after preprocessing, to avoid edge effects."""
        if cutoff==None:
            self.cutoff=0
        elif cutoff>=0:
            self.cutoff = cutoff
        else:
            print(print_terminal(type='error',message='Cutoff size should be >= 0'))
            raise ValueError('Cutoff size should be >= 0')
        
        if not ml_in_use: #make them empty in case if preprocessed run twice(due to appends)
            dataset.ft_extracted         = []
            dataset.processed_data       = []
            dataset.processed_timestamps = []
            dataset.head_dir             = []
            dataset.speed                = []
            dataset.position             = []
        
        
    def set_cpu_gpu_mode(self,use_gpu, use_paralell):
        """
        It will sets the program to use or not GPU or paralell processing based 
        on the computers parameters in order to reach the most optimal speed.
        Based on the given parameters it will set the self.cpu_gpu_mode:
            = 1 --> the program will try to use GPU
            = 2 --> the program will try to use CPU with paralell processing
            = 3 --> the program will try to use CPU WITHOUT paralell processing
        ----------
        Args:
            use_gpu (bool): if you want to use the GPU set it True.
            use_paralell (bool): if you want to use paralell processing set it True.
        """
        
        if use_paralell:
            os.environ['SSQ_PARALLEL'] = '1'
            self.is_paral              = True
        else: 
            os.environ['SSQ_PARALLEL'] = '0'
            self.is_paral              = False
            
        if use_gpu and self.gpu:
            try: #it will try to use make_scale function, because it sometimes fales on GPU due to unknown reasons
                make_scales(10, 1, 2, nv=2, scaletype='log-piecewise', wavelet='morlet', downsample=1)
                self.cpu_gpu_mode = 1 #use GPU
                os.environ['SSQ_GPU'] = '1'
                os.environ['SSQ_PARALLEL'] = '0'
                print(print_terminal(type='run',message='Use GPU: True'))
            except:
                self.gpu = False
                print(print_terminal(type='run',message='Use GPU: False'))
        else:
            self.gpu = False
            print(print_terminal(type='run',message='Use GPU: False'))
        
        if not self.gpu:
            if self.count_cpu >10: #if the computer has more than 10 CPU cores the paralell processing will be efficient enough to use it
                self.cpu_gpu_mode = 2 #use CPU with paralell
                os.environ['SSQ_GPU'] = '0'
                print(print_terminal(type='run',message='Paralell computing with CPU: True'))
            else: #otherwise don't use paralell process.
                self.cpu_gpu_mode = 3 #use CPU without paralell
                os.environ['SSQ_GPU'] = '0'
                print(print_terminal(type='run',message='Paralell computing with CPU: False'))
            
    
    def get_output_sizes(self, dataset):
        """
        Get needed sizes for preprocessing.
        ----------
        Args:
            dataset (obj): dataset with the raw signal.
        Returns:
            num_chunks (int): how many chunks in the whole tetrode signal lenght
            output_len (int): size of the output signal length per chunk
            num_channels (int): number of good channels in raw signal
            win_per_chunk (int): number of windows in a chunk
        """
        
        win_per_chunk   = self.chunk_size // self.window_size
        if self.saver:
            self.chunk_size = win_per_chunk * self.window_size #save remain part in a chunk which is <window_size
        data_len        = dataset.raw_signal.shape[0]
        num_chunks      = data_len//self.chunk_size
        remain_windows  = (data_len-(num_chunks*win_per_chunk*self.window_size))//self.window_size
        
        return num_chunks, remain_windows, win_per_chunk
    
    
    def set_srate(self, dataset, tasks, rewrite = False):
        """
        If during preprocessing the new sampling rate is different than the original,
        checks if it will not upsampling and if it's smaller and rewrite= True it 
        sets the srate to the new sampling rate. 
        ----------
        Args:
            dataset (obj): the dataset of the signal
            tasks (dict): the preprocessing steps and parameters.
            rewrite (bool, optional): Rewrite the original srate for the new one. Defaults to False.
                                      Set it True, only at the end of preprocessing!!!
        Returns:
            task(dict): it gives back the task file. If new srate>=orig. srate it sets downsampling False.
        """
        try:
            if tasks['down_samp'][0]<dataset.tetrode_srate:
                if rewrite:
                    dataset.tetrode_srate = tasks['down_samp'][0]
            elif tasks['down_samp'][0]>dataset.tetrode_srate:
                print(print_terminal(type='error',message='New sampling rate bigger than the original'))
                raise ValueError("New sampling rate bigger than the original")
            else:
                raise
        except:
            tasks['down_samp'] = False
        
        return tasks
    
    def smooth_signal(self, signal, N):
        """
        Simple smoothing by convolving a filter with 1/N.
        ----------
        DeepInsight Toolbox
        © Markus Frey
        https://github.com/CYHSM/DeepInsight
        Licensed under MIT License
        ----------
        Args:
            signal (array_like): Signal to be smoothed
            N (int): smoothing_factor
        Returns:
            signal (array_like): Smoothed signal
        """
        # Preprocess edges
        signal = np.concatenate([signal[0:N], signal, signal[-N:]])
        # Convolve
        signal = np.convolve(signal, np.ones((N,))/N, mode='same')
        # Postprocess edges
        signal = signal[N:-N]

        return signal

    
    def calculate_speed_from_position(self, positions, interval, smoothing=False):
        """
        Calculate speed from X,Y coordinates.
        ----------
        DeepInsight Toolbox
        © Markus Frey
        https://github.com/CYHSM/DeepInsight
        Licensed under MIT License
        ----------
        Args:
            positions ((N, 2) array_like): N samples of observations, containing X and Y coordinates
            interval (int): Duration between observations (in s, equal to 1 / sr)
            smoothing (bool or int, optional):  If speeds should be smoothed, by default False/0
        Returns:
            speed ((N, 1) array_like): Instantenous speed of the animal
        """
        X, Y = positions[:, 0], positions[:, 1]
        # Smooth diffs instead of speeds directly
        Xdiff = np.diff(X)
        Ydiff = np.diff(Y)
        if smoothing:
            Xdiff = self.smooth_signal(Xdiff, smoothing)
            Ydiff = self.smooth_signal(Ydiff, smoothing)
        speed = np.sqrt(Xdiff**2 + Ydiff**2) / interval
        speed = np.append(speed, speed[-1])

        return speed
    
    
    def calculate_head_direction_from_leds(self, positions, return_as_deg=False):
        """
        Calculates head direction based on X and Y coordinates with two LEDs.
        ----------
        DeepInsight Toolbox
        © Markus Frey
        https://github.com/CYHSM/DeepInsight
        Licensed under MIT License
        ----------
        Args:
            positions ((N, 2) array_like): N samples of observations, containing X and Y coordinates
            return_as_deg (bool): Return heading in radians or degree
        Returns:
            head_direction ((N, 1) array_like): Head direction of the animal
        """
        
        X_led1, Y_led1, X_led2, Y_led2 = positions[:, 0], positions[:, 1], positions[:, 2], positions[:, 3]
        # Calculate head direction
        head_direction = np.arctan2(X_led1 - X_led2, Y_led1 - Y_led2)
        # Put in right perspective in relation to the environment
        offset = +np.pi/2
        head_direction = (head_direction + offset + np.pi) % (2*np.pi) - np.pi
        head_direction *= -1
        if return_as_deg:
            head_direction = head_direction * (180 / np.pi)

        return head_direction
    
    @staticmethod
    @nb.njit(parallel=True)
    def resize_extracted(signal_batch, new_width, new_height):
        """
        It resize the wavelet transformed windows into a smaller size to save memory.
        ----------
        Source:
        https://stackoverflow.com/questions/55275466/numpy-resize-3d-array-with-interpolation-in-2d
        ----------
        Args:
            signal_batch (3d array): preprocessed 
            new_width (int): width of the output cwt coefficients
            new_height (int): height of the output cwt coefficients

        Returns:
            _type_: _description_
        """
        dtype = signal_batch.dtype
        signal_batch = signal_batch.T
        n, width, height = signal_batch.shape[:3]
        w = np.empty(new_width, dtype=dtype)
        for i in range(new_width):
            w[i] = (width - 1) * i / (new_width - 1)
        h = np.empty(new_height, dtype=dtype)
        for i in range(new_height):
            h[i] = (height - 1) * i / (new_height - 1)
        ii_1 = w.astype(np.int32)
        ii_2 = np.minimum(ii_1 + 1, width - 1)
        w_alpha = w - ii_1
        w_alpha_1 = 1 - (w_alpha)
        jj_1 = h.astype(np.int32)
        jj_2 = np.minimum(jj_1 + 1, height - 1)
        h_alpha = h - jj_1
        h_alpha_1 = 1 - (h_alpha)
        out = np.empty((n, new_width, new_height) + signal_batch.shape[3:], dtype=dtype)
        for idx in nb.prange(n):
            for i in nb.prange(new_width):
                for j in nb.prange(new_height):
                    out_11 = signal_batch[idx, ii_1[i], jj_1[j]]
                    out_12 = signal_batch[idx, ii_1[i], jj_2[j]]
                    out_21 = signal_batch[idx, ii_2[i], jj_1[j]]
                    out_22 = signal_batch[idx, ii_2[i], jj_2[j]]
                    out_1 = out_11 * h_alpha_1[j] + out_12 * h_alpha[j]
                    out_2 = out_21 * h_alpha_1[j] + out_22 * h_alpha[j]
                    out[idx, i, j] = out_1 * w_alpha_1[i] + out_2 * w_alpha[i]
        return out.T
    
    
    def convert_LEDs(self, srate, tetrode_timestamps, led_timestamps, raw_led_positions):
        """
        1. Downscale and interpolate the LED postions to 1 coordinate per window
        2. Calculate the head direction, speed, and position of the animal
        ----------
        DeepInsight Toolbox
        © Markus Frey
        https://github.com/CYHSM/DeepInsight
        Licensed under MIT License
        ----------
        Args:
            dataset (obj): Dataset of the raw signal.
            tetrode_timestamps(vector): Tetrode's timestamps
            led_timestamps (vector): Timestamps of LEDs positions
            raw_led_positions (vector): Position of leds (4 colums(x1,y1,x2,y2))
        """
        
        # Get coordinates of both LEDs
        raw_timestamps = tetrode_timestamps[()]
        
        output_x_led1 = np.interp(raw_timestamps[np.arange(0, raw_timestamps.shape[0],
                                                        self.window_size)], led_timestamps, raw_led_positions[:, 0])
        output_y_led1 = np.interp(raw_timestamps[np.arange(0, raw_timestamps.shape[0],
                                                        self.window_size)], led_timestamps, raw_led_positions[:, 1])
        output_x_led2 = np.interp(raw_timestamps[np.arange(0, raw_timestamps.shape[0],
                                                        self.window_size)], led_timestamps, raw_led_positions[:, 2])
        output_y_led2 = np.interp(raw_timestamps[np.arange(0, raw_timestamps.shape[0],
                                                        self.window_size)], led_timestamps, raw_led_positions[:, 3])
        raw_positions = np.array([output_x_led1, output_y_led1, output_x_led2, output_y_led2]).transpose()

        # Clean raw_positions and get centre
        positions_smooth = pd.DataFrame(raw_positions.copy()).interpolate(
                                        limit_direction='both').rolling(5, min_periods=1).mean().values
        position = np.array([(positions_smooth[:, 0] + positions_smooth[:, 2]) / 2,
                            (positions_smooth[:, 1] + positions_smooth[:, 3]) / 2]).transpose()

        # Also get head direction and speed from positions
        speed = self.calculate_speed_from_position(position, interval=1/(srate//self.window_size), smoothing=3)
        head_direction = self.calculate_head_direction_from_leds(positions_smooth, return_as_deg=False)

        return head_direction,speed,position
        
        
    def chebyBandpassFilter(self, signal, lowcut, highcut, fs=30000, order=5):
        """
        Bandpass filter the signal with a Chebyshev filter.
        ----------
        Args:
            signal (array): input signal.Can be 1 or 2 dimensional.
            lowcut (int): lowcut value. Normaly 2Hz
            highcut (int): hightcut value. Normaly 250Hz
            fs (int, optional): sampling frequency in Hz. Defaults to 30000.
            order (int, optional): order of the filter. Defaults to 5.

        Returns:
            signal(array): preprocessed signal.Can be 1 or 2 dimensional.
        """

        b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
        filteredSignal = lfilter(b, a, signal)
        
        return filteredSignal


    def notch_filter(self, signal, notch_freq=50, quality_factor=20, fs=30000):
        """
        Notch filter to filter electrical line noise.
        ----------
        Args:
            signal (array): input signal.Can be 1 or 2 dimensional.
            notch_freq (int, optional): Frequency of the electrical system in Hz. Defaults to 50.
            quality_factor (int, optional): filtering quality. Defaults to 20.
            fs (int, optional): sampling frequency in Hz. Defaults to 30000.

        Returns:
            signal(array): preprocessed signal.Can be 1 or 2 dimensional.
        """
        
        b_notch, a_notch = iirnotch(notch_freq, quality_factor, fs)
        return filtfilt(b_notch, a_notch, signal)
    
    
    def downsample(self, signal, timestamps, oldFS=30000, newFS=20000, nwind=None):
        """
        Dowsampling the original signal to save memory space and help the training.
        ----------
        Args:
            signal (array): input signal.Can be 1 or 2 dimensional.
            timestamps (array): timestamps of the raw signal.
            oldFS (int, optional): Original sampling rate in Hz. Defaults to 30000.
            newFS (int, optional): Downsampled target sampling rate in Hz. Defaults to 20000.
            nwind (int, optional): The number of windows into how many parts the prepared signal should be cut later. Defaults to None.

        Returns:
            signal(array): preprocessed signal.Can be 1 or 2 dimensional.
        """
        
        newData=[]
        xnew=[]
        #if our signal has >1 channels and we want to downsample it by windows
        if nwind!=None and len(signal.shape)==2:
            #it iterates through the input signal and downsamples it by the size of the window
            for i in range(nwind):
                start=i*self.window_size
                stop=start+self.window_size
                sign_part = signal[start:stop,:]
                ts_part = timestamps[start:stop]
                newNumSamples = int((sign_part.shape[0] / oldFS) * newFS)
                newData.append(resample(sign_part, newNumSamples))
                xnew.append(np.linspace(ts_part[0], ts_part[-1], newNumSamples, endpoint=False))
            return np.concatenate(xnew), np.concatenate(newData)
        elif nwind==None or len(signal.shape)==1:
            #if the signal needed to process as one chunk or there it has only 1 channels -->this used during dimension check
            newNumSamples = int((signal.shape[0] / oldFS) * newFS)
            newData = resample(signal, newNumSamples)
            xnew = np.linspace(timestamps[0], timestamps[-1], newNumSamples, endpoint=False)
            return xnew, newData
    
    
    def detrend(self, signal):
        """
        Filter out linear trends from the signal.
        ----------
        Args:
            signal (array): input signal.Can be 1 or 2 dimensional.
        Returns:
            signal(array): preprocessed signal.Can be 1 or 2 dimensional.
        """

        return detrend(signal)
    
    
    def rolling_mean(self, signal, wind_fact=None):
        """
        Calculate the mean of the signal by windows. This rolling windows size is usually different from the
        preprocess window size.
        ----------
        Args:
            signal (array): input signal.Can be 1 or 2 dimensional.
            wind_fact(int): size of the rolling window's size. Default to None.

        Returns:
            signal(array): preprocessed signal.Can be 1 or 2 dimensional.
        """

        return np.squeeze(pd.DataFrame(signal).rolling(wind_fact,axis=0, min_periods=1).mean().values)
    
    
    def normalize(self, signal):
        """
        Normalizing chunk values to between [-1,1]

        Args:
            signal (array): The 1D or 2D input signal.

        Returns:
            (numpy array): normalized signal
        """
        
        # Only this is changed to use 2-norm put 2 instead of 1
        norm = np.linalg.norm(signal)
        # normalized matrix
        signal = signal/norm  
        return signal
    
    def set_wt_cwt_mode(self,mode):
        """
        Returns with the class of the choosed cwt wavelet methode.
        ----------
        Args:
            mode (str): name of the wavelet. Option(s): morlet --TODO: add more options

        Returns:
            wavelet class: returns with the right wavelet class
        """
        
        if mode=='morlet':
            return Morlet()
        else:
            print(print_terminal(type='error',message='This wavelet mode has not been developed yet.'))
            raise ValueError('This wavelet mode has not been developed yet.')
    
    
    def create_scales(self, epoch_len, low_lim, high_lim, nv=6, scaletype='log-piecewise', wavelet='morlet'):
        """
        It returns with the frequency scale array of the wavelet transformation.
        ----------
        Args:
            epoch_len (int): length of the input signal.
            low_lim (int): lowcut value of the wavelet transform in Hz. Normaly 2
            high_lim (_type_): highcut value of the wavelet transform in Hz. Normaly 250
            nv (int, optional): Number of voices (wavelets) per octave. Defaults to 6.
            scaletype (str, optional): Scaling methode.Options: ['log', 'log-piecewise', 'linear']. Defaults to 'log-piecewise'.
            wavelet (str, optional): Name of the wavelet. Used only for scaletype='log-piecewise'. Defaults to 'morlet'.

        Returns:
            scales(array): list of frequencies.
        """
        
        return make_scales(epoch_len, low_lim, high_lim, nv=nv, scaletype=scaletype, wavelet=wavelet)
    
    
    def cwt(self, signal, scales, mode = "morlet"):
        """
        It does the continious wavelet transfrom on one chunk, on the possibly fastest way.
        ----------
        Args:
            signal (array): input signal.Can be 1 or 2 dimensional.
            scales (array): scales of frequencies from the create_scales function.
            mode (str, optional): wavelet name. Defaults to "morlet".

        Returns:
            coeffs(array): absolute values of coefficients by the size (num of frequecies, len of timestamps, num of channels)
            freqs (array): frequencies. Identical with the input scales.
        """
  
        if self.cpu_gpu_mode==1: #If using gpu
            
            try: #if there is any problem with the gpu processing it changes the processing type to CPU without paralell
                if len(signal.shape)==2: #if input signal 2D
                    coeffs = []
                    for ch in range(signal.shape[1]): #it processes the channels one by one
                        coeffs_, freqs_ = ss_cwt(signal[:,ch], wavelet=Wavelet(), scales=scales, cache_wavelet=True)
                        coeffs.append(np.abs(coeffs_.cpu().numpy())) #read coeffs from gpu ram
                        freqs = freqs_.cpu().numpy() #read freqs from gpu ram
                        del coeffs_,freqs_; gc.collect() #clear them from the gpu ram
                    coeffs = np.swapaxes(np.array(coeffs).T,0,1) #convert to the right output shape
                    
                elif len(signal.shape)==1: #if input signal 1D
                    coeffs_, freqs_ = ss_cwt(signal, wavelet=Wavelet(), scales=scales, cache_wavelet=True)
                    coeffs = np.abs(coeffs_.cpu().numpy())
                    freqs = freqs_.cpu().numpy()
                    del coeffs_,freqs_; gc.collect()
            except:
                print(print_terminal(type='error',message="There is not enough memory for run on GPU. It's continue to run on CPU. If you want to run on GPU, set smaller scale range."))
                self.cpu_gpu_mode==3
            
        elif self.cpu_gpu_mode==2 and self.is_paral: #If using cpu and paralell computing
            mode = self.set_wt_cwt_mode(mode) #get wavelet class
            if signal.shape[1]==len(self.good_ch): #get the list of channels to process
                good_ch = self.good_ch
            elif len(signal.shape)==1:
                good_ch = [0]
            else:
                good_ch = range(signal.shape[1])
            
            #paralell comp. set the number of jobs= number of CPU cores//5 --> based on experimenting it gives the fastest result.
            result = Parallel(n_jobs=self.count_cpu//5)(delayed(wt_cwt)(signal[:,ch],mode,scales) for ch in good_ch)
            coeffs = np.reshape(result,(-1,len(signal),len(result))) #convert to the right output size
            coeffs = np.abs(coeffs)
            freqs  = scales

        elif self.cpu_gpu_mode==3:
            mode = self.set_wt_cwt_mode(mode)
            if len(signal.shape)==2: #if the signal is 2D
                coeffs = []
                for ch in range(signal.shape[1]): #iterating through the channels and processing them
                    coeffs.append(np.abs(wt_cwt(signal[:,ch], wavelet=mode, widths=scales)))
                coeffs = np.swapaxes(np.array(coeffs).T,0,1) #convert to to right output size
            elif len(signal.shape)==1: #if singal is 1D
                coeffs = np.abs(wt_cwt(signal, wavelet=mode, widths=scales))
            freqs = scales
            
        return coeffs, freqs
        
    
    def process_window(self, 
                       input_signal, 
                       timestamps_out, 
                       tasks, 
                       srate, 
                       win_per_chunk=None, 
                       dim_check=False):
        
        """
        Preprrocess one chunk and slice them into windows.
        ----------
        Args:
            input_signal(array): signal 1 or 2 dimensional.
            timestamps_out(array): timestamps of the signal.
            tasks(dict): required preprocessing steps and parameters.
            srate(int): original sampling rate of the signal.
            win_per_chunk(int, optional): number of windows in a chunk. Default to None.
            dim_check(bool, optional): if it is True returns only with the dimensions of the processed signal.Default to False.
            
        Returns:
            (array): if dim_check is True it returns with the shape of the processed wavelet transformed or simple signal based on the tasks.
            If dim_check is False, it returns with the processed signal, the feature extracted signal (or an empty list is there is no task as ft_extr)
            and with the timestamps of the processed signal.
        """
        
        ft_extr_out = []
        orig_srate  = srate
        
        #if dim_check=True load only one channel
        if dim_check:
            signal_out = input_signal[:,0]
        else:
            signal_out = input_signal
            
        #iterate through the tasks
        for taskID in tasks.keys():
            
            if not dim_check: 
                scales = make_scales(30000, 2, 250, nv=5, scaletype='log-piecewise', wavelet='morlet')

            #----
            if taskID == 'Cheby_band':
                if tasks[taskID] != False:
                    #lowcut, highcut, order=5
                    signal_out = self.chebyBandpassFilter(signal_out, tasks[taskID][0], tasks[taskID][1], fs=orig_srate, order=tasks[taskID][2])
            #----
            if taskID == 'narrow_filt':
                if tasks[taskID] != False:
                    #notch_freq=50,quality_factor=20
                    signal_out = self.notch_filter(signal_out, notch_freq=tasks[taskID][0], quality_factor=tasks[taskID][1], fs=orig_srate)
            #----
            if taskID == 'down_samp':
                if tasks[taskID] != False:
                    #new_srate =20000
                    timestamps_out, signal_out = self.downsample(signal_out, timestamps_out, oldFS=orig_srate, newFS=tasks[taskID][0], nwind=win_per_chunk)
                    orig_srate = tasks[taskID][0]
            #----
            if taskID == 'detrend':
                if tasks[taskID] != False:
                    signal_out = self.detrend(signal_out)
            #----
            if taskID == 'roll_mean':
                if tasks[taskID] != False:
                    #wind_fact=None
                    signal_out = self.rolling_mean(signal_out, tasks[taskID][0])
            #----
            if taskID == 'normalize':
                if tasks[taskID] != False:
                    signal_out = self.normalize(signal_out)
            #----
            if taskID == 'feature_extract':
                if tasks[taskID] != False:
                    #low_lim, high_lim, nv=6, scaletype='log-piecewise', wavelet='morlet', rescale_size=64
                    scales = self.create_scales(len(timestamps_out),
                                                low_lim=tasks[taskID][0], 
                                                high_lim=tasks[taskID][1], 
                                                nv=tasks[taskID][2], 
                                                scaletype=tasks[taskID][3], 
                                                wavelet=tasks[taskID][4])
                    ft_extr_out, *_ = self.cwt(signal_out, scales,  mode=tasks[taskID][4])
        
                    if len(signal_out.shape)==2: #if signal is 2D
                        if not self.cutoff ==0: #if cutoff>0 cut off the first and last few ms of the signall
                            ft_extr_out = ft_extr_out[:,self.cutoff:-self.cutoff,:]
                            signal_out = signal_out[self.cutoff:-self.cutoff,:]
                            timestamps_out = timestamps_out[self.cutoff:-self.cutoff]
                        if win_per_chunk!=None or win_per_chunk!=0: #if it is >1 windows in a chunk reshape it
                            ft_extr_out = np.swapaxes(np.reshape(ft_extr_out,(ft_extr_out.shape[0],win_per_chunk,-1,ft_extr_out.shape[2])),0,1)    
                            signal_out = np.reshape(signal_out, (win_per_chunk,-1,signal_out.shape[1]))
                            timestamps_out = np.reshape(timestamps_out, (win_per_chunk,-1))  
                        if not tasks[taskID][5]==0:
                            ft_extr_out = self.resize_extracted(ft_extr_out, tasks[taskID][5], tasks[taskID][5])
                    elif len(signal_out.shape)==1: #if signal 1D
                        if not self.cutoff ==0:
                            ft_extr_out = ft_extr_out[:,self.cutoff:-self.cutoff]
                            signal_out = signal_out[self.cutoff:-self.cutoff]
                            timestamps_out = timestamps_out[self.cutoff:-self.cutoff]
                        if not tasks[taskID][5]==0:
                            ft_extr_out = resize(ft_extr_out, (tasks[taskID][5], tasks[taskID][5]), mode = 'constant')
        
                    
        if dim_check: #if dim_check=True return with dimensions
            if tasks.get('feature_extract') is not None:
                return (np.array(ft_extr_out)).shape
            else:
                return (np.array(signal_out)).shape
        else:
            # else return with processed feature extracted 2D matrix, signal and processed timestamps
            return (np.array(ft_extr_out), np.array(signal_out), np.array(timestamps_out))


    def preprocess_chunks(self, dataset, tasks, is_test=False, live_write=False, hdf5_file=None):
        """
        It slices the signal into chunks and process it one by one with process_window feature. Finally it refresh the dataset.
        ----------
        Args:
            dataset (obj): The dataset of the raw signal
            tasks (dict): required preprocessing steps and parameters.
            is_test (bool): if it's True, process only 1 chunk.
            live_write (bool): if it's True the processed data will be saved during the preprocessing. Default to False.
            hdf5_file (h5py obj): if live_write True, the preprocess_pipeline gives the output file.
        """
        
        num_chunks, remain_windows, win_per_chunk = self.get_output_sizes(dataset)        
        chunk_start = 0

        #if if saving is and there is any last part of signal which is shorter than a chunk add a last iteration
        if self.saver and remain_windows>0:
            num_ch_proc = num_chunks+1
        else:
            num_ch_proc = num_chunks
        if is_test:    
            num_ch_proc = 1 #if testing

        #iterate through chunks and show progressbar
        for chunk_idx in trange(num_ch_proc, desc='Processing chunks'):
            
            #get the end of the first chunk
            if chunk_idx==num_chunks: #if there is only 1 chunk
                chunk_stop = len(dataset.tetrode_timestamps)
                win_per_chunk = remain_windows
            else:
                chunk_stop = chunk_start + self.chunk_size + self.cutoff
            
            raw_chunk   = np.array(dataset.raw_signal[chunk_start:chunk_stop,self.good_ch])
            time_chunk  = np.array(dataset.tetrode_timestamps[chunk_start:chunk_stop])

            (ft_extr,
             processed_sign, 
             processed_timestamps)= self.process_window(input_signal=raw_chunk,
                                                        timestamps_out=time_chunk, 
                                                        tasks=tasks,
                                                        srate=dataset.tetrode_srate,
                                                        win_per_chunk=win_per_chunk,
                                                        dim_check=False) 
             
            if live_write:
                #save the calculated data into the output file
                DataFile().create_or_update(hdf5_file, "outputs/ft_extracted", ft_extr.shape, np.float64, ft_extr)
                DataFile().create_or_update(hdf5_file, "outputs/processed_data", processed_sign.shape, np.float64, processed_sign)
                DataFile().create_or_update(hdf5_file, "outputs/processed_timestamps", processed_timestamps.shape, np.float64, processed_timestamps)
            else:
                #add processed windows to dataset
                dataset.ft_extracted.append(ft_extr)
                dataset.processed_data.append(processed_sign)
                dataset.processed_timestamps.append(processed_timestamps)    
                chunk_start = chunk_stop - self.cutoff
                
        if not live_write:
            #Delete unnecessary dimensions
            dataset.ft_extracted = np.concatenate(dataset.ft_extracted)
            dataset.processed_data = np.concatenate(dataset.processed_data)
            dataset.processed_timestamps = np.concatenate(dataset.processed_timestamps)
        else:
            #load the calculated parameters for training phase
            dataset.ft_extracted = hdf5_file["outputs/ft_extracted"]
            dataset.processed_data = hdf5_file["outputs/processed_data"]
            dataset.processed_timestamps = hdf5_file["outputs/processed_timestamps"]


def preprocess_pipeline(dataset, 
                        tasks, 
                        file_path, 
                        window_size=2000,
                        chunk_size=30000, 
                        is_gpu= False, 
                        is_paral=True,
                        cutoff=0,
                        load_code=0,
                        use_saver=True,
                        is_test=False,
                        live_write=False):
    """
    Handle the whole preprocessing.
    ----------
    Args:
        dataset (obj): The dataset of the raw signal
        tasks (dict): Required preprocessing steps and parameters.
        file_path (str): Output file path
        window_size (int, optional): Size of one window. Defaults to 2000.
        chunk_size (int, optional): Size of one chunk. Defaults to 30000.
        is_gpu (bool, optional): Set True if you want to use GPU during preprocessing. Defaults to False.
        is_paral (bool, optional): Set True if you want to use paralell during preprocessing. Defaults to True.
        cutoff (int, optional): Size of the part that it will cut off from the start and the end of the chunk to avoid edge effects.
                                Defaults to 0.
        load_code (int): Define the input data. If =0 there is raw data. Default to 0.
        use_saver(bool): If it's True the progam will find the chunk size with the minimum data surplus
        is_test (bool): If it's True, won't be saved any result and only 1 chunk will be processed.
        live_write (bool): If it's True the processed data will be saved during the preprocessing. 
                           This mode saves RAM, but slows the preprocessing.Default to False.
    """
    
    if load_code==0:
        #TODO: add items check to tasks check
        ID_list = ['Cheby_band','narrow_filt','down_samp','detrend','roll_mean','normalize','feature_extract']
        ID_bools=[id in ID_list for id in tasks.keys()]
        if not all(ID_bools):
            error_ids=[x for x, y in zip(list(tasks.keys()), ID_bools) if not y]
            print(print_terminal(type='error', message='problem with the following key name(s) in tasks: {0}. Check them and run preprocessing again.'.format(error_ids)))
        
        prep = Preprocess(dataset,
                        window_size=window_size,
                        chunk_size=chunk_size,
                        is_gpu=is_gpu,
                        is_paral=is_paral,
                        cutoff=cutoff,
                        ml_in_use=False,
                        use_saver=use_saver)
        print(print_terminal(type='done',message='Preprocessing function initialized.'))
        
        #If new srate bigger than the original, set the new srate as original srate and downsample
        tasks = prep.set_srate(dataset, tasks, rewrite=False)
        
        if live_write:
            folder= '/'.join(file_path.split('/')[:-1])+'/'
            format= pathlib.Path(file_path).suffix
            if len(format)==0:
                fname= str(file_path.split('/')[-1])
            else:
                fname= str(file_path.split('/')[-1][:-len(format)])
            file_path = '{0}processed/processed_{1}{2}'.format(folder,fname,'.h5')
            DataFile().write_processed_h5_file(path=file_path, dataset=dataset)
            hdf5_file = h5py.File(file_path, mode='a')
            print(print_terminal(type='run',message='Continuous writing during preprocessing is on.'))

        #3.Test output size
        output_dims = prep.process_window( 
                                        input_signal=dataset.raw_signal[0:prep.window_size,:],
                                        timestamps_out=dataset.tetrode_timestamps[0:prep.window_size],
                                        tasks=tasks,
                                        srate=dataset.tetrode_srate,
                                        dim_check=True)
        
        #4.Process chunks
        print(print_terminal(type='run',message='Preprocessing has been started...'))
        if live_write:
            prep.preprocess_chunks(dataset,tasks, is_test=is_test, live_write=live_write, hdf5_file=hdf5_file)
        else:
            prep.preprocess_chunks(dataset,tasks, is_test=is_test, live_write=live_write)
        prep.set_srate(dataset, tasks, rewrite=True)
        dataset.preprocessed = True
        dataset.preprocess_steps_values = tasks
        
        #5.Save it    
        while True:
            if is_test:
                answer='n'
                print(print_terminal(type='run',message='Test mode is active. No file will be saved and only 1 chunk will be processed.'))
            elif live_write:
                answer='n'
                print(print_terminal(type='done',message='Continuous writing sucessfully finished.'))
            else:
                answer = input("Do you want to save the preprocessed file?: (Y/N)").lower()
                print(print_terminal(type='inp',message='Save preprocessed file?',answer=answer))
                
            if (answer == 'y') or (answer == 'yes'):
                folder= '/'.join(file_path.split('/')[:-1])+'/'
                format= pathlib.Path(file_path).suffix
                if len(format)==0:
                    fname= str(file_path.split('/')[-1])
                else:
                    fname= str(file_path.split('/')[-1][:-len(format)])
                file_path = '{0}processed/processed_{1}{2}'.format(folder,fname,'.h5')
                DataFile().write_processed_h5_file(path=file_path, dataset=dataset)
                print(print_terminal(type='done',message='Preprocessing finished, and results are saved.'))
                break
            elif (answer == 'n') or (answer == 'no'):
                print(print_terminal(type='done',message='Preprocessing finished without saving the results.'))
                break
            else:
                print(print_terminal(type='error',message='\n Please answere with Y/Yes or N/no'))
    else:
        print(print_terminal(type='error',message='Data already preprocessed. Jump to training.'))
        
    
    
