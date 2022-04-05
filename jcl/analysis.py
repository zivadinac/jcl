import numpy as np
from scipy.ndimage import gaussian_filter


def __calc_num_bins(maze_size, bin_size):
    """ Calculate number of spatial bins.

        Args:
            maze_size - size of the maze
            bin_size - size of a bin
        Return:
            number of bins, same shape as maze_size
    """
    return 1 + (np.array(maze_size) / bin_size).astype(int)


def __get_bin_idx(p, bin_size):
    """ Calculate bin index for given position.

        Args:
            p - position
            bin_size - size of bins
        Return:
            bin index - tuple of the same shape as p
    """
    return tuple((p // bin_size).astype(int))


def occupancy_map(positions, maze_size, bin_size, bin_len, smooth_sd=None):
    """ Produce occupancy map.

        Args:
            positions - array of positions of shape (n,1) or (n,2)
            maze_size - size of the maze in the same units as positions
            bin_size - size of spatial bins in the same units as positions
            bin_len - duration of temporal bins in ms
            smooth_sd - SD in bins for gaussian smoothing
        Return:
            occupancy - time in seconds spent in each spatial bin
    """
    occupancy = np.zeros(__calc_num_bins(maze_size, bin_size))

    for p in positions:
        bin_idx = __get_bin_idx(p, bin_size)
        occupancy[bin_idx] += bin_len

    if smooth_sd is not None:
        occupancy = gaussian_filter(occupancy, smooth_sd)

    return occupancy / 1000 # ms to seconds


def place_fields(spike_train, positions, maze_size, bin_size, bin_len, smooth_sd=3, return_occupancy=False):
    """ Produce place field map for the given single cell spike train.

        Args:
            positions - array of positions of shape (n,1) or (n,2)
            maze_size - size of the maze in the same units as positions
            bin_size - size of spatial bins in the same units as positions
            bin_len - duration of temporal bins in ms
            smooth_sd - SD in bins for gaussian smoothing
            return_occupancy - whether to return occupancy map
        Return:
            place field map - matrix with firing rate (Hz) in each spatial bin
    """
    assert len(spike_train) == len(positions)
    occupancy = occupancy_map(positions, maze_size, bin_size, bin_len, smooth_sd)
    pfs = np.zeros_like(occupancy)

    for  p, sn in zip(positions, spike_train):
        bin_idx = __get_bin_idx(p, bin_size)
        pfs[bin_idx] += sn

    pfs = pfs / occupancy
    pfs[np.isnan(pfs)] = 0
    pfs[np.isinf(pfs)] = 0
    pfs = gaussian_filter(pfs, smooth_sd)
    return pfs, occupancy if return_occupancy else pfs


def frs(place_fields, eps=1e-15):
    """ Compute mean, median and max firing rates from given place field map.
        Ignore nans, infs and values smaller than eps.

        Args:
            place_fields - map with firing rates for spatial bins
            eps - precision, ignore bins with values smaller than eps
        Return:
            (mean, median, max) FR in Hz
    """
    not_nan = np.logical_not(np.isnan(place_fields))
    not_inf = np.logical_not(np.isinf(place_fields))
    good = np.logical_and(not_nan, not_inf)
    good = np.logical_and(good, place_fields >= eps)
    pfs = place_fields[good]
    return pfs.mean(), np.median(pfs), pfs.max()


def place_field_sharpness(place_fields, occupancy, eps=1e-15):
    """ Compute sharpness of given place field (in bits/spike).

        Args:
            place_fields - map with firing rates for spatial bins
            occupancy - map with time spent in each spatial bin
            eps - precision, ignore bins with values smaller than eps
        Return:
            sharpness in bits/spike
    """
    assert place_fields.shape == occupancy.shape
    occ_prob = occupancy / occupancy.sum()
    mfr = frs(place_fields, eps)[0]

    per_bin = []
    for fr, op in zip(place_fields.flatten(), occ_prob.flatten()):
        if np.isnan(fr) or np.isnan(op) or\
           np.isinf(fr) or np.isinf(op) or\
           fr < eps or op < eps:
            continue
        i = fr / mfr
        per_bin.append(i * np.log2(i) * op)
    return np.sum(per_bin)
    
