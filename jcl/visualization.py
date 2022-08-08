import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from_list = LinearSegmentedColormap.from_list
from matplotlib.cm import ScalarMappable
from jcl.utils import trial_distance, trial_duration
from jcl.analysis import Map


def plot_map(m: Map, title=None, colorbar_label=None, path=None):
    """ Plot given with added color bar."

        Args:
            m - map
            title - figure title
            colorbar_label - label to be used for the colorbar
            path - file path to which to save the figure, if none show the figure
    """
    if m.ndim == 1:
        im = np.expand_dims(m.map, 1)
        im = np.repeat(im.map, 3, 1).T
    else:
        im = m.map

    plt.set_cmap("jet")
    plt.imshow(im)

    cb = plt.colorbar(ticks=[0., np.max(im)])
    if colorbar_label is not None:
        cb.set_ticklabels([0, np.round(np.max(im))])
        cb.set_label("Hz", fontsize=18)

    if title is not None:
        plt.title(title)

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()


def plot_trajectory(trajectory, size, sampling_rate, title=None, path=None, trajectory_2=None):
    """ Plot given trajectory.

        Args:
            trajectory - animal trajectory during a trial, array of shape (n,2)
            size - figure size, tuple
            sampling_rate - number of samples per second (Hz)
            title - figure title
            path - file path to which to save the figure, if none show the figure
            trajectory_2 - second animal trajectory during a trial, array of the same shape as trajectory
    """
    im = np.ones((size[0]+1, size[1]+1, 3))

    trajectory = trajectory.astype(np.int16)
    for i in range(len(trajectory)):
        cc = i / len(trajectory)
        color = (cc, 0., cc)
        p = [trajectory[i][0], trajectory[i][1]]
        im[p[0], p[1]] = color

    if trajectory_2 is not None:
        trajectory_2 = trajectory_2.astype(np.int16)
        for i in range(len(trajectory_2)):
            p = [trajectory_2[i][0], trajectory_2[i][1]]
            im[p[0], p[1]] = (0., 1., 0.)

    plt.imshow(im, vmin=0., vmax=1.)
    plt.xticks([])
    plt.yticks([])
    cmap = ScalarMappable(cmap=from_list('time_cmap', [(0., 0., 0.), (1., 0., 1.)]))
    cb = plt.colorbar(cmap, ticks=[0., 1.])
    cb.set_ticklabels([0, np.round(trial_duration(trajectory, sampling_rate), 2)])
    cb.set_label("Time (s)", fontsize=18)

    if title is not None:
        plt.title(title, fontsize=24)

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()


def plot_trials_in_session(trial_inds, sampling_rate=None, path=None):
    """ Visualize trials in within recording session.

        Args:
            trial_inds - list of trial indices (list of tuples)
            sampling_rate - whl sampling rate, if None (default) time won't be converted to seconds
            path - path to which save the image
    """
    if sampling_rate is not None:
        trial_inds = [(t[0] / sampling_rate, t[1] / sampling_rate) for t in trial_inds]
        plt.xlabel("Time (s)")

    for i, t in enumerate(trial_inds):
        plt.fill_betweenx([0, 1], t[0], t[1], color="orange", alpha=.45)
        txt = f"Trial {i+1}"
        if sampling_rate is not None:
            txt += f" ({np.round(t[1]-t[0], 2)}s)"
        plt.text((t[0]+t[1])/2, .5, txt, rotation="vertical")

    plt.yticks([])

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


"""
# quick test
# TODO unit tests

plot_place_field(np.array([0,0,0,1,2,3,2,1,0,0,0,0,0,0]))
plot_place_field(np.array([[0,0,0,1,2,3,2,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,2,3,2,1,0,0,0]]))
"""
