import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from jcl.utils import trial_distance

def plot_place_field(place_field):
    """ Plot place field with added color bar."

        Args:
            place_field - place field as np.array
    """
    if place_field.ndim == 1:
        im = np.expand_dims(place_field, 1)   
        im = np.repeat(im, 3, 1).T
        plt.yticks([])
        plt.xticks(np.arange(place_field.shape[0]))
    else:
        im = place_field
        plt.yticks(np.arange(place_field.shape[0]))
        plt.xticks(np.arange(place_field.shape[1]))

    plt.imshow(im)
    plt.colorbar()
    plt.show()

def plot_trajectory(xy_trajectory, size, title=None, sampling_rate=40):
    xy_trajectory = xy_trajectory.astype(np.int16)
    im = np.ones((size[0], size[1], 3))

    for i in range(len(xy_trajectory)):
        color = (i / len(xy_trajectory), 0., 0.)
        p = [xy_trajectory[i][1], xy_trajectory[i][0]]
        im[p[0], p[1]] = color

    plt.imshow(im, vmin=0., vmax=1.)
    cmap = ScalarMappable(cmap=LinearSegmentedColormap.from_list('time_cmap', [(0.,0.,0.), (1.,0.,0.)]))
    cb = plt.colorbar(cmap, ticks=[0., 1.])
    cb.set_ticklabels([0, np.round(trial_distance(xy_trajectory), 2)])
    cb.set_label("Distance covered", fontsize=18)
    if title is not None:
        plt.title(title, fontsize=24)
    plt.show()


"""
# quick test
# TODO unit tests

plot_place_field(np.array([0,0,0,1,2,3,2,1,0,0,0,0,0,0]))
plot_place_field(np.array([[0,0,0,1,2,3,2,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,2,3,2,1,0,0,0]]))
"""
