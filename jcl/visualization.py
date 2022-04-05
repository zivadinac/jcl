import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from_list = LinearSegmentedColormap.from_list
from matplotlib.cm import ScalarMappable
from jcl.utils import trial_distance, trial_duration

def plot_place_field(place_field, title=None):
    """ Plot place field with added color bar."

        Args:
            place_field - place field as an np.array
    """
    if place_field.ndim == 1:
        im = np.expand_dims(place_field, 1)
        im = np.repeat(im, 3, 1).T
    else:
        im = place_field

    plt.set_cmap("jet")
    plt.imshow(im)
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()

def plot_trajectory(xy_trajectory, size, sampling_rate, title=None):
    """ Plot given trajectory.

    Args:
        xy_trajectory - animal trajectory during a trial, array of shape (n,2)
        size - figure size, tuple
        sampling_rate - number of samples per second (Hz)
        title - figure title
    """
    xy_trajectory = xy_trajectory.astype(np.int16)
    im = np.ones((size[0]+1, size[1]+1, 3))

    for i in range(len(xy_trajectory)):
        cc = i / len(xy_trajectory)
        color = (cc, 0., cc)
        #p = [xy_trajectory[i][1], xy_trajectory[i][0]]
        p = [xy_trajectory[i][0], xy_trajectory[i][1]]
        im[p[0], p[1]] = color

    plt.imshow(im, vmin=0., vmax=1.)
    plt.xticks([])
    plt.yticks([])
    cmap = ScalarMappable(cmap=from_list('time_cmap', [(0., 0., 0.), (1., 0., 1.)]))
    cb = plt.colorbar(cmap, ticks=[0., 1.])
    cb.set_ticklabels([0, np.round(trial_duration(xy_trajectory, sampling_rate), 2)])
    cb.set_label("Time (s)", fontsize=18)
    if title is not None:
        plt.title(title, fontsize=24)
    plt.show()


"""
# quick test
# TODO unit tests

plot_place_field(np.array([0,0,0,1,2,3,2,1,0,0,0,0,0,0]))
plot_place_field(np.array([[0,0,0,1,2,3,2,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,2,3,2,1,0,0,0]]))
"""
