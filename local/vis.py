import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import torchaudio
import numpy as np


import pdb
test_token = [{'word': 'MANY', 'start_time': 1.3, 'end_time': 1.5}, {'word': 'COMPLICATED', 'start_time': 1.56, 'end_time': 2.14}, {'word': 'IDEAS', 'start_time': 2.24, 'end_time': 2.56}, {'word': 'ABOUT', 'start_time': 2.66, 'end_time': 2.9}, {'word': 'THE', 'start_time': 3.0, 'end_time': 3.06}, {'word': 'RAINBOW', 'start_time': 3.14, 'end_time': 3.42}, {'word': 'HAVE', 'start_time': 3.48, 'end_time': 3.58}, {'word': 'BEEN', 'start_time': 3.62, 'end_time': 3.74}, {'word': 'FORMED', 'start_time': 3.84, 'end_time': 4.16}]
laronix_green = [120, 189, 145]


def token_plot(audio, sr, token):
    # pdb.set_trace()
    # Get X axis
    duration = audio.squeeze().shape[0] / sr
    x = np.arange(0, duration, 1/sr)
    # Wave plot
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(x, audio.squeeze(), color="#78bd91")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Amplitude")
    
    y_limit = np.max(audio.numpy()) 
    # pdb.set_trace()
    # load token 
    for i in token:
        word, start_time, end_time = i.values()
        # plot tokens
        ax.text(x=start_time, y=y_limit, s=word, ha="left", fontsize="large", fontstretch="ultra-condensed")
        # plot token boundarys
        ax.vlines(x=start_time, ymin=np.min(audio.numpy()), ymax=y_limit, colors="black")
        # ax.vlines(x=end_time, ymin=np.min(audio.numpy()), ymax=y_limit, colors="red")

    
    plt.tight_layout()        
    # pdb.set_trace()
    # fig.savefig("1.png")
    return fig
    