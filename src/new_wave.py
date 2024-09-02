import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import pylab
# imports for debugging purposes
import readline
import code

# Moving average filter
def moving_average(x, win_size):
    filtered = np.convolve(x, np.ones(win_size), 'valid') / win_size
    filtered = np.append(np.zeros(win_size-1), filtered) # fill the w-1 first slots with zeros
    return filtered

# Median filter
def median_filter(x, win_size):
    S = 1
    nrows = ((x.size-win_size)//S)+1
    n = x.strides[0]
    strided = np.lib.stride_tricks.as_strided(x, shape=(nrows,win_size), strides=(S*n,n))
    filtered  = np.median(strided,axis=1)
    filtered = np.append(np.zeros(win_size-1), filtered) # fill the w-1 first slots with zeros
    return filtered

def butterworth_lowpass_filter(data, cutoff_freq, fs, order=2):
    #fs is the sampling rate
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # lfilter apply filter along one dimension
    y = lfilter(b, a, data)
    return y

def filter_data(data):    
    plt.figure(figsize=(10,6))
    plt.plot(data)
    # Moving average with 21-day window
    filtered_data = moving_average(data, 14)
    plt.plot(filtered_data, label='ma')

    # 2nd Order Low-Pass Filter with 14-day window
    order = 2
    fs = len(data) # sampling rate       
    cutoff = 21 # cutoff freq.
    filtered_data =  butterworth_lowpass_filter(filtered_data, cutoff, fs, order)
    plt.plot(filtered_data, label='bw')

    # Median filter with 14-day window
    filtered_data = median_filter(filtered_data, 14)

    plt.plot(filtered_data, label='me')

    # Reduce the delay effect introduced by the filtering process
    # Advance the signal by 25 days
    filtered_data = filtered_data[25:]

    plt.plot(filtered_data, label='shift')
    plt.legend()
    plt.show()

    return filtered_data

# Forward Euler Approximation (Euler's Method)
def forward_euler(t, step, y, dy0):
    dy = np.zeros(len(t))
    dy[0] = dy0
    for i in range(len(t) - 1):
        dy[i+1] = (y[i+1] - y[i])/step
    return dy

# New epidemiological wave detection
#   Receives (sec_der, abs_threshold)
#      sec_der: second derivative of acc. number of cases 
#      abs_threshold: threshold to consider around zero
#   Returns (x_t, y_t) -> coordinates of the transition points 
def new_wave_detection(sec_der, abs_threshold):
    x_t = []
    y_t = []
    for i in range(len(sec_der)-1):
        if((sec_der[i] < abs_threshold) and (sec_der[i+1] > abs_threshold)):
           x_t.append(i+1)
           y_t.append(sec_der[i+1])
    return x_t, y_t

# Get transition points
#  Receives (data): accumulated indicator
#  optional (visual): show transition points graph ?
#  optional (city_name): city name 
#  optional (threshold): threshold 
def get_transition_points(data, visual=False, city_name = "", threshold = 3e-5, indicator = 'cases'):    
    
    plt.rcParams.update({'font.size': 18})

    # Normalize by maximum value
    normalized_acc_n_cases = data / max(data)

    t = np.linspace(0, len(normalized_acc_n_cases), len(normalized_acc_n_cases))
    daily_n_cases = forward_euler(t, 1, normalized_acc_n_cases, 0)

    # Filter data to reduce noise effects
    unf_daily_n_cases = daily_n_cases
    daily_n_cases = filter_data(daily_n_cases)

    # Obtain second derivative of the number of cases w.r.t time
    # using Forward Euler
    t = np.linspace(0, len(daily_n_cases), len(daily_n_cases))
    sd0 = daily_n_cases[1] - daily_n_cases[0]
    sec_der = forward_euler(t, 1, daily_n_cases, sd0)

    # Detection of new waves
    abs_threshold = threshold
    x_t, y_t = new_wave_detection(sec_der, abs_threshold)

    if(visual):        
        # Graph with acc. data and its first two derivatives
        fig, axs = plt.subplots(3, 1, figsize=(10,14)) # 3 rows, 1 col
        plt.tight_layout(pad=1.5)
        #plt.suptitle(f"{city_name} threshold {abs_threshold} ", fontsize=16)
        plt.suptitle(f"{city_name}", fontsize=24)
        axs[0].plot(normalized_acc_n_cases) # para alinhar as retas de nova onda
        axs[0].vlines(x_t, 1, 3e-4, colors='dimgray', linestyles='dashdot', zorder=1, label="new wave transition")
        axs[0].set_title(f'Normalized accumulated number of {indicator}')
        axs[0].set_ylabel(f"${indicator}$")

        axs[1].plot(unf_daily_n_cases, c='darkgray')
        axs[1].plot(daily_n_cases)
        axs[1].ticklabel_format(axis='y',style='sci',scilimits=(-2,-2))
        axs[1].vlines(x_t, min(unf_daily_n_cases), max(unf_daily_n_cases), colors='dimgray', linestyles='dashdot', zorder=1, label="new wave transition")
        axs[1].set_title('First derivative')
        axs[1].set_ylabel(f"${indicator}$ / $day$")

        axs[2].ticklabel_format(axis='y',style='sci',scilimits=(-4,-4))
        axs[2].set_ylim(-2e-4, 2e-4)
        axs[2].set_title("Second derivative - New wave detection")
        axs[2].set_xlabel("t (days)")
        axs[2].set_ylabel(f"${indicator}$ / $day^2$")
        axs[2].plot(sec_der, zorder=1) # obs: check if this scaling is correct
        axs[2].hlines([-1*abs_threshold, abs_threshold], 0, len(sec_der), colors='silver', linestyles='dashed', zorder=1, label=f"threshold = $\pm${abs_threshold}")
        axs[2].vlines(x_t, -3e-4, 3e-4, colors='dimgray', linestyles='dashdot', zorder=1, label="new wave transition")
        axs[2].scatter(x_t, y_t, s=15, c='r', zorder=2, label="sign change in the second derivative")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'Figuras/{city_name}_nw', facecolor='white', dpi=200)
        plt.show(block=False)

        # return series data to plot ouside this function
        tran_pts_fig_series_data = {
            'normalized_acc_n_cases': normalized_acc_n_cases,
            'unf_daily_n_cases':unf_daily_n_cases,
            'daily_n_cases':daily_n_cases,
            'sec_der':sec_der,
            'abs_threshold':abs_threshold,
            'x_t':x_t,
            'y_t':y_t
        }
    return x_t, fig, tran_pts_fig_series_data
