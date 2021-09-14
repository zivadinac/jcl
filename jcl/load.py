import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from jcl.utils import get_last_spike_time

def __to_ms(x, sampling_period):
    """ Convert time `x` to ms.

        Args:
            x - time to be converted (number or array of numbers)
            sampling_period - sampling period in ms
        Return:
            Converted data of the same type as `x`
    """
    return x * sampling_period

def readfromtxt(file_path, conv_fun=str):
    """ Read lines from a text file into a list.

        Args:
            file_path - path to the text file
            conv_fun - conversion to be done on a line
        Return:
            List of elements converted using conv_fun
    """
    with open(file_path) as f:
        lines = [conv_fun(l) for l in f]
    return lines


def spike_times_from_res_and_clu(res_path, clu_path, exclude_noise_clusters=True):
    """ Load spike times for each neuron from provided '.res' and '.clu' files. Doesn't include clusters without spikes.

        Args:
            res_path - path to res file (sorted in a non-descending order)
            clu_path - path to clu file
            exclude_noise_clusters - by default don't return spikes belonging to noise clusers (0 - artifacts and 1 - unassigned spikes)
        Return:
            List of firing times for each cell (list of np.arrays)
    """
    # .clu and .res files are big
    # so we use our faster implementation
    # instead of np.genfromtxt
    clu = np.array(readfromtxt(clu_path, conv_fun=int))
    res = np.array(readfromtxt(res_path, conv_fun=int))
    assert len(clu) == len(res) or len(clu) == len(res) + 1

    if len(clu) == len(res) + 1:
        # number of clusters is written in the first line
        clu_num = clu[0]
        clu = clu[1:]
    else:
        clu_num = clu.max()

    first_clu = 2 if exclude_noise_clusters else 0
    spike_times = [res[clu == i] for i in range(first_clu, clu_num + 1)]
    spike_times = list(filter(lambda l: len(l) > 0, spike_times))

    return spike_times


def slice_spike_times(spike_times, begin_ts, end_ts):
    """ Return only spike times between given timestamps.

        Args:
            spike_times - list of spike times per neuron (list of iterables, pre-sorted in a non-descending order)
            begin_ts - spike times greater or equal than this
            end_ts - spikes times less than this

        Return:
            Sliced spike times (list of np.arrays)
    """
    if begin_ts <= 0 and end_ts >= get_last_spike_time(spike_times):
        # no need to iterate over spike_times
        # if the given range contains all of them
        return spike_times
    return [st[np.logical_and(st >= begin_ts, st < end_ts)] for st in spike_times]


def bins_from_spike_times(spike_times, bin_len=25.6, sampling_period=0.05, dtype=np.uint16, dense_loading=True, return_mat_type=csc_matrix):
    """ Bin given spike times, each bin contains total number of spikes.

        Args:
            spike_times - list of spike times per neuron (list of iterables, pre-sorted in a non-descending order)
            bin_len - length of a bin in ms (default 1s/39.0625 = 25.6ms)
            sampling_period - sampling period in ms (default 1s/20kHz = 0.05ms)
            dtype - dtype to use for bins, default np.uint16 (np.uint8 would use less memory, but can store only up to 256 spikes per bin)
            dense_loading - if True (default) load data into a dense np.ndarray then convert to a `return_mat_type` (fast), if False load into a sparse matrix (slow, but memory efficient)
            return_mat_type - type of matrix to be returned (default is sparse `csc_matrix` for efficient storage and relatively fast column slicing)
        Return:
            Matrix with spike count per bin ((neuron num, time bins), `return_mat_type`)
    """
    # leave this two lines here in case we find non-sorted spikes (.res files)
    # maxes = [np.max(st) if len(st) > 0 else 0 for st in spike_times]
    # last_spike_time = __to_ms(np.max(maxes), sampling_period)  # in ms
    last_spike_time = np.max([__to_ms(st[-1], sampling_period) for st in spike_times])
    bin_num = np.ceil(last_spike_time / bin_len).astype(int)
    bin_edges = np.arange(bin_num + 1) * bin_len

    if dense_loading:
        # fastest loading (due to indexing), but requires a lot of memory
        binned_data = np.empty((len(spike_times), bin_num), dtype=np.uint16)
    else:
        # lil_matrix is fast for loading rows, convert later to return_mat_type
        binned_data = lil_matrix((len(spike_times), bin_num), dtype=np.uint16)

    for n, st in enumerate(spike_times):
        st_ms = __to_ms(np.array(st), sampling_period)
        hist = np.histogram(st_ms, bins=bin_edges)[0]
        binned_data[n] = hist

    return return_mat_type(binned_data)


def bins(res_path, clu_path, bin_len=25.6, sampling_period=0.05, dtype=np.uint16, dense_loading=True, return_mat_type=csc_matrix):
    """ Bin spike times given '.res' and '.clu' files, each bin contains total number of spikes (without noise clusters).

        Args:
            res_path - path to res file
            clu_path - path to clu file
            bin_len - length of a bin in ms (default 1s/39.0625 = 25.6ms)
            sampling_period - sampling period in ms (default 1s/20kHz = 0.05ms)
            dtype - dtype to use for bins, default np.uint16 (np.uint8 would use less memory, but can store only up to 256 spikes per bin)
            dense_loading - if `return_dense` is True this is ignored and only dense np.ndarray will be used; otherwise: if True (default) load data into a dense array then convert to a sparse matrix (fast), if False load straight into a sparse matrix (requires less memory, but significantly slower)
            return_mat_type - if `return_dense` is True this is ignored; otherwise: type of sparse matrix to be returned (default `csc_matrix` for fast column slicing)
        Return:
            Matrix of shape (neuron num, time bins) with spike count in each bin (np.array or csc_matrix, depending on `return_dense`)
    """
    st = spike_times_from_res_and_clu(res_path, clu_path)
    return bins_from_spike_times(st, bin_len, sampling_period, dense_loading, return_mat_type)


def cell_types(des_path):
    """ Read cell types from '.des' file.
            'p1' - CA1 pyramidal
            'pu' - unknown pyramidal
            'b1' - CA1 interneuron
            'bu' - unknown interneuron

        Args:
            des_path - path to des file
        Return:
            Array with cell types (dtype=str)
    """
    return np.genfromtxt(des_path, dtype=str)


def positions_from_whl(whl_path):
    """ Load animal positions from '.whl' file.

        Args:
            whl_path - path to whl file
        Return:
            Array with positions (N, led * 2)
    """
    return np.genfromtxt(whl_path)


def positions_and_speed_from_whl2(whl2_path):
    """ Load animal positions and speed from '.whl2' file.

        Args:
            whl2_path - path to whl2 file
        Return:
            Array with positions (N, 2), array with speed (N)
    """
    d = np.genfromtxt(whl2_path)
    return d[:, [0, 1]], d[:, 2]


def session_limits(resofs_path):
    """ Read session limits from a .resofs file.

        Args:
            resofs_path - path to the .reso file
        Return:
            session limits (list of tuples)
    """
    resofs = readfromtxt(resofs_path, int)
    num_sessions = len(resofs)
    resofs = [0] + resofs
    return [(resofs[i], resofs[i+1]) for i in range(num_sessions)]

# TODO unit tests


"""
st = [np.array([1,3,5,7,9,11]), np.array([2,4,6,8,10,12,14])]
sts = slice_spike_times(st, 4, 14)
"""
