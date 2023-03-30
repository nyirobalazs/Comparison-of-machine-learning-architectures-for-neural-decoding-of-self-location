import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import read_write
print(read_write.print_terminal(type='done',message='Python package initialization done in visualization.'))   

def find_nearest(timevector, timepoint):
    """
    Find the nearest timestamp in an array to the given timepoint
    ----------
    Args:
        timevector (array): the timevector array with timestamps
        timepoint (float): the reference timepoint
    Returns:
        idx (int): the index of the closest timepoint in timevector array
    """
    
    timevector = np.asarray(timevector) # Convert the input to an array.
    idx = (np.abs(timevector - timepoint)).argmin()  #find closest distance

    return idx

def plot_raw_tetrode(timestamps,
                    tetrode_data, 
                    plotted_ch=5, 
                    horizontal_shift=800,
                    title="Raw tetrode channels",
                    is_save = False,
                    load_code=0,
                    save_path = './tetrode_raw.svg'):
    
    """
    Create a plot about the tetrode signal.
    ----------
    Args:
        timestamps (numpy vect): tetrode timevector (x axis)
        tetrode_data (numpy vect): tetrode signal data (y axis)
        plotted_ch (int): the first x channel(S) that will be plotted
        horizontal_shift (int): the horizontal space between the channels
        titles (array): the titles of the plot. (eg. ['a'])
        load_code (int): define the input data. If =0 there is raw data. Default to 0.
        save_path (str): the plot will be saved to this filepath if is_save=True. Default to './tetrode_raw.svg'.
    """
    if load_code==0:
        # Create a subplot with 1 row and 3 columns 
        fig = go.Figure()

        # Creating the tetrode chart. Iterate through the channels and plot them
        for i in range(1,plotted_ch+1):
            fig.add_trace(go.Scatter(x=list(timestamps), y=list(tetrode_data[:,i]+(i*horizontal_shift))))
        
        # Plot settings
        fig.update_traces(showlegend=False)
        fig.update_xaxes(showgrid=False) # Don't show the grids
        fig.update_yaxes(showticklabels=False) # Don't show stickers
        fig.update_layout(height=500, 
                        width=700,
                        title=title,
                        xaxis_title='Time (ms)',
                        yaxis_title='Channels (mV)',
                        xaxis=dict(rangeslider=dict(visible=True),))

        fig.show()
        
        if is_save:
            read_write.save_plot(path=save_path,fig=fig)
            print(read_write.print_terminal(type='done',message='Tetrode signal plotted and saved to: {}'.format(save_path)))
        else:
            print(read_write.print_terminal(type='done',message='Tetrode signal plotted without saving it.'))
    else:
        print(read_write.print_terminal(type='error',message='No raw data due to preprocessed input data.'))


def plot_raw_data(timestamps,
                  tetrode_data, 
                  head_direction,
                  speed, 
                  position,
                  plotted_ch=5, 
                  horizontal_shift=800,
                  conv_2_degree=True,
                  titles=["Raw tetrode channels","Head direction and speed","Position"],
                  save_path='./images/raw_tetr.svg',
                  is_save=True):
    
    """
    Create a plot about the tetrode signal, speed and head direction(polar coord) and the positions in the 2D space.
    ----------
    Args:
        timestamps (numpy vect): tetrode timevector (x axis)
        tetrode_data (numpy vect): tetrode signal data (y axis)
        head_direction (numpy vect): head direction in radian/degrees
        positons (numpy vect): positions in the 2D space
        plotted_ch (int): the first x channel(S) that will be plotted
        horizontal_shift (int): the horizontal space between the channels
        conv_2_degree (bool): if True the radian input values will be converted into degrees
        titles (array): the titles of the three plots. (eg. ['a', 'b', 'c'])
        save_path (str): the plot will be saved to this filepath if is_save=True. Default to './images/raw_tetr.svg'.
        is_save (bool): if True, the plot will be saved.
    """
    
    # Convert radian values into degrees
    if conv_2_degree:
        head_direction = head_direction*(180/np.pi)

    # Create a subplot with 1 row and 3 columns 
    fig = make_subplots(rows=1, cols=3,
                        specs=[[{},{"type": "scatterpolar"},{}]],
                        subplot_titles=(titles[0], titles[1], titles[2]),
                        horizontal_spacing = 0.0001
                        )
    # Creating the tetrode chart. Iterate through the channels and plot them
    for i in range(1,plotted_ch+1):
        fig.add_trace(go.Scatter(x=list(timestamps), y=list(tetrode_data[:,i]+(i*horizontal_shift))),
                      row=1, col=1
                      )
    # Plot the head directions and the speed on a polar coordinate
    fig.add_trace(
            go.Scatterpolar(r=speed[:,0], theta=head_direction[:,0],
                            subplot = "polar",
                            mode = 'markers'),
                            row=1, col=2
                            )
    # Plot the positions
    fig.add_trace(go.Scatter(x=list(position[:,0]), y=list(position[:,1])),
                  row=1, col=3
                 )
    # Plot settings
    fig.update_traces(showlegend=False)
    fig.update_xaxes(title_text='Time', row=1, col=1) # x axis' name of the 1st plot
    fig.update_yaxes(title_text='Channels', row=1, col=1) # y axis' name of the 1st plot
    fig.update_xaxes(title_text='X', row=1, col=3) 
    fig.update_yaxes(title_text='Y', row=1, col=3) 
    fig.update_xaxes(showgrid=False) # Don't show the grids
    fig.update_yaxes(showgrid=False)
    fig.update_yaxes(showticklabels=False, row=1, col=1) # Don't show stickers
    fig.update_annotations(y=1.15, selector={'text':titles[0]}) # The space beetween the title and the 1st plot 
    fig.update_annotations(y=1.15, selector={'text':titles[1]})
    fig.update_annotations(y=1.13, selector={'text':titles[2]})
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True),)) # slider under 1st plot
    fig.update_layout(polar = dict(radialaxis_range = [-1*max(speed[:,0]), max(speed[:,0])])) #range of the speed array on 2nd plot
    fig.show()

    if is_save:
        read_write.save_plot(path=save_path,fig=fig)
        print(read_write.print_terminal(type='done',message='Raw tetrode signal plotted and saved to: {}'.format(save_path)))
    else:
        print(read_write.print_terminal(type='done',message='Raw tetrode signal plotted without saving it.'))
    

def plot_cwt(time, coefficients, frequencies, is_save = False, save_path = './images/cwt.svg'):
    """
    It plots the scalogram of the cwt's.
    ----------
    Args:
        time (array): time array of the scalogram
        coefficients (numpy array): a array with the absolute values of the wavelet transformet signal's coefficients
        frequencies (numpy array): the frequencies of the wavelet transform
        save_path (str): the plot will be saved to this filepath if is_save=True. Default to './images/cwt.svg'.
        is_save (bool): if True, the plot will be saved.
    """
    #calculate the power and the period of the cofficients
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies

    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, np.log2(period), np.log2(power))

    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)
    
    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.show()

    plt.figure(figsize=(15.5, 8))
    plt.imshow(np.abs(coefficients), aspect='auto')
    
    if is_save:
        read_write.save_plot(fig, save_path)
        print(read_write.print_terminal(type='done',message='Wavelet transformed result(s) plotted and saved to: {}'.format(save_path)))
    else:
        print(read_write.print_terminal(type='done',message='Wavelet transformed result(s) plotted without saving it.'))
    
    
def processed_plot(timestamps,
                   raw_tetrode,                   
                   processed_tetrode,
                   titles=['Raw and processed signal','Preprocessed'],
                   convert_ms = True,
                   is_save = False,
                   load_code = 0,
                   save_path = './tetrode_raw.svg'):
    """
    It create a plot about the raw and preprocessed signals in the given sequence.
    ----------
    Args:
        timestamps (array): timestamps of the raw signal.
        raw_tetrode (array): raw signal part.
        processed_tetrode (array): processed signal part.
        titles (list, optional): Titles of the subplots. Defaults to ['Raw and processed signal', 'Preprocessed']
        convert_ms (bool): if it's True time will be converted from seconds to milliseconds.
        is_save (bool, optional): If you want to save the file set it True. Defaults to False.
        load_code (int): define the input data. If =0 there is raw data. Default to 0.
        save_path (str, optional): saving filepath. Defaults to './tetrode_raw.svg'.
    """
    
    if load_code==0:
        if convert_ms:
            time_raw = timestamps/1000 #convert milliseconds to seconds
        else:
            time_raw = timestamps
            
        raw      = raw_tetrode
        proc     = processed_tetrode

        fig = make_subplots(rows=2, cols=1,
                        column_widths=[1],
                        shared_yaxes=False,
                        vertical_spacing = 0.1,
                        specs=[[{"type": "scatter"}],[{"type": "scatter"}]],
                        subplot_titles=(titles[0], titles[1]))

        fig.add_trace(go.Scatter(x=time_raw, y=raw),row=1, col=1)
        fig.add_trace(go.Scatter(x=time_raw, y=proc),row=2, col=1)

        # Plot the head directions and the speed on a polar coordinate
        fig.update_xaxes(showgrid=False) # Don't show the grids
        fig.update_yaxes(showgrid=False) # Don't show the grids
        fig.update_xaxes(showticklabels=False,row=1, col=1) 
        fig.update_xaxes(showticklabels=True,row=2, col=1)
        fig.update_xaxes(title_text='Time (s)', row=2, col=1) 
        fig.update_yaxes(title_text='mV', row=1, col=1)
        fig.update_yaxes(title_text='', row=2, col=1)
        fig.update_traces(showlegend=False)
        fig.update_layout(height=500, width=1400, title_text="Tetrode data before and after pre-processing")
        fig.update_coloraxes(showscale=False)
        fig.show()
        
        if is_save:
            read_write.save_plot(path=save_path, fig=fig)
            print(read_write.print_terminal(type='done',message='Processed signal plotted and saved.'))
        else:
            print(read_write.print_terminal(type='done',message='Processed signal plotted without saving it.'))
    else:
        print(read_write.print_terminal(type='error', message='Preprocessed data were loaded, do not have raw data to do this plot. Jump to training.'))
      
            
def plot_speed_and_head(df, 
                        titles = ['Distribution of speeds', 'Distribution of head directions'],
                        is_save = False,
                        save_path = './sp_hd_distr.svg'):

    """
    Plot only the speeds and head direction distributions.
    ----------
    Args:
        titles(list,otpional): titles of subplots. Default to ['Distribution of speeds', 'Distribution of head directions'].
        is_save (bool, optional): If you want to save the file set it True. Defaults to False.
        save_path (str, optional): saving filepath. Defaults to './sp_hd_distr.svg'.
    """
    fig = make_subplots(rows=2, cols=1,
                     vertical_spacing = 0.1,
                     subplot_titles=(titles[0], titles[1]),
                     specs=[[{"type": "histogram"}],[{"type": "histogram"}]])
    
    fig.add_trace(px.histogram(df, x="speed", nbins=20),row=1, col=1)
    fig.add_trace(px.histogram(df, x="head_dir", nbins=20),row=2, col=1)
    fig.show()
       
    if is_save:
        read_write.save_plot(fig, save_path)
        print(read_write.print_terminal(type='done',message='Processed speed(s) and head direction(s) plotted and saved.'))
    else:
        print(read_write.print_terminal(type='done',message='Processed speed(s) and head direction(s) plotted without saving it.'))
        
        
def plot_cumulative_distribution(bins_count, cumulative, pdf, save_path='./images/cumulative.svg', is_save=False):
    """
    Plot cumulative distribution vs. PDF and vs. CDF of the trained network.
    ----------
    Args:
        save_path (str): path of the output saving path. Default to './images/cumulative.svg'
        bins_count (int): number of bins
        cumulative (array): cummulative distribution values.
        pdf (array): pdf values.
        is_save (bool, optional): If you want to save the plot set it True. Defaults to False.
    """
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bins_count[1:], y=pdf,
                        mode='lines',
                        name='PDF'))
    fig.add_trace(go.Scatter(x=bins_count[1:], y=cumulative,
                        mode='lines',
                        name='CDF'))
    fig.update_layout(title='Cumulative Distribution',
                    xaxis_title='Error (cm)',
                    yaxis_title='Cumulative fraction')
    fig.show()
    
    if is_save:
        read_write.save_plot(fig, save_path)
        print(read_write.print_terminal(type='done',message='Cumulative distribution plotted and saved.'))
    else:
        print(read_write.print_terminal(type='done',message='Cumulative distribution plotted without saving it.'))