from collections import Counter
import numpy as np


def __majority_filter(array, width):
    """ Replace every element within neighborhood of width `width` with it's most common element. """
    offset = width // 2
    array = [0] * offset + array
    return [Counter(a).most_common(1)[0][0]
        for a in (array[i:i+width] for i in range(len(array) - offset))]
 
 
def in_area(trajectory, corners):
    """ Find points in the trajectory that are withing the given area.

        Args:
            trajectory - array of shape (N, 2)
            corners - top-left and bottom-right corners of the area,
                      list of tuples: [(TL_vert, TL_hor), (BR_ver, BR_hor)]

        Return:
            Boolean array of shape (N,)
    """
    x_idx = (corners[0][0] <= trajectory[:, 0]) & (trajectory[:, 0] <= corners[1][0])
    y_idx = (corners[0][1] <= trajectory[:, 1]) & (trajectory[:, 1] <= corners[1][1])
    return x_idx & y_idx


def get_trial_idx(trajectory, sb=None, maze=None, dur_thr=2.5*50):
    """ Get indices of trials in the given full-session trajectory.

        Args:
            trajectory - array of shape (N, 2)
            sb - top-left and bottom-right corners of the start box,
                 list of tuples: [(TL_vert, TL_hor), (BR_ver, BR_hor)]
            maze - top-left and bottom-right corners of the maze,
                   list of tuples: [(TL_vert, TL_hor), (BR_ver, BR_hor)]
            dur_thr - minimal trial duration (in number of samples)

        Return:
            Trial indices (list of tuples)
    """
    if sb is None and maze is None:
        raise ValueError("Please provide start box limits (`sb`) or\
                          maze corners (`maze`).")

    if sb is not None and maze is not None:
        raise ValueError("Please provide either start box limits (`sb`) or\
                          maze corners (`maze`), but not both.")

    if sb is not None:
        sb_inds = in_area(trajectory, sb)
        trial_inds = np.logical_not(sb_inds).tolist()

    if maze is not None:
        trial_inds = in_area(trajectory, maze).tolist()

    trial_inds = __majority_filter(trial_inds, 50)
    trial_borders = []
 
    for i in range(1, len(trial_inds)-1):
        if trial_inds[i] == 1 and trial_inds[i-1] == 0:
            trial_borders.append(i)
        elif trial_inds[i] == 1 and trial_inds[i+1] == 0:
            trial_borders.append(i)

    if len(trial_borders) % 2 == 1:
        trial_borders = trial_borders[:-1]

    return [(trial_borders[i], trial_borders[i+1])
            for i in range(0, len(trial_borders), 2)
            if trial_borders[i+1]-trial_borders[i] >= dur_thr]
