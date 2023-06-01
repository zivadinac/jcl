import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from_list = LinearSegmentedColormap.from_list
from matplotlib.cm import ScalarMappable
from jcl.utils import trial_distance, trial_duration
from jcl.analysis import Map
import plotly.express as px
import plotly.graph_objects as go


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


def plot_trajectory(trajectory, size, sampling_rate, title=None, path=None, ax=None, trajectory_2=None, colorbar=True):
    """ Plot given trajectory.

        Args:
            trajectory - animal trajectory during a trial, array of shape (n,2)
            size - figure size - tuple (unused - kept only for compatibility, will be removed)
            sampling_rate - number of samples per second (Hz)
            title - figure title
            path - file path at which to save the figure; if `None` and ax is `None` show the figure
            ax - axes object for plotting (plotly figure); if `None` creates a new one
            trajectory_2 - second animal trajectory during a trial, array of the same shape as trajectory
            colorbar - whether to plot color bar; default True

        Return:
            Figure with plotted trajectory.
    """

    duration = len(trajectory) / sampling_rate
    color = np.arange(len(trajectory)) / len(trajectory) * duration
    fig = px.scatter(x=trajectory[:, 0], y=trajectory[:, 1],
                     color=color, range_color=(0, duration))

    if ax is not None:
        t = list(fig.select_traces())[0]
        ax.add_trace(t)
        fig = ax

    if colorbar:
        fig.update_layout(coloraxis_colorbar_title_text = 'Time (s)')
    else:
        fig.update(layout_coloraxis_showscale=False)

    if ax is None:
        fig.update_xaxes(showgrid=False, visible=False)
        fig.update_yaxes(showgrid=False, visible=False)
        fig.update_layout(template="plotly_white")
        # if figure is provided we don't want to change its style

    if trajectory_2 is not None:
        t2t = go.Scatter(x=trajectory_2[:, 0], y=trajectory_2[:, 1],
                         mode="lines", showlegend=False, line_color="green")
        fig.add_trace(t2t)

    if title is not None:
        fig.update_layout(title=title)

    if path is not None:
        fig.write_image(path)
    if ax is None and path is None:
        fig.show()
    return fig


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
