import numpy as np

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
    """ Load spike times for each neuron from provided '.res' and '.clu' files. Doesn't include noise clusters (0 and 1).
        
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

    return spike_times

def bins_from_spike_times(spike_times, bin_len=25.6, sampling_period=0.05, dtype=np.uint16):
    """ Bin given spike times, each bin contains total number of spikes.

        Args:
            spike_times - list of spike times per neuron (list of iterables, pre-sorted in a non-descending order)
            bin_len - length of a bin in ms (default 1s/39.0625 = 25.6ms)
            sampling_period - sampling period in ms (default 1s/20kHz = 0.05ms)
            dtype - dtype to use for bins, default np.uint16 (np.uint8 would use less memory, but can store only up to 256 spikes per bin)
        Return:
            Matrix (neuron num, time bins) of with spike count in each bin
    """
    # leave this two lines here in case we find non-sorted spikes (.res files)
    # maxes = [np.max(st) if len(st) > 0 else 0 for st in spike_times]
    # last_spike_time = __to_ms(np.max(maxes), sampling_period)  # in ms
    last_spike_time = np.max([__to_ms(st[-1], sampling_period) for st in spike_times])

    bin_num = np.ceil(last_spike_time / bin_len).astype(int)
    bin_edges = np.arange(bin_num + 1) * bin_len

    binned_data = np.empty((len(spike_times), bin_num), dtype=np.uint16)
    for n, st in enumerate(spike_times):
        st_ms = __to_ms(np.array(st), sampling_period)
        hist = np.histogram(st_ms, bins=bin_edges)[0]
        binned_data[n] = hist

    return binned_data

def bins(res_path, clu_path, bin_len=25.6, sampling_period=0.05, dtype=np.uint16):
    """ Produce population vectors for given '.res' and '.clu' files.

        Args:
            res_path - path to res file
            clu_path - path to clu file
            bin_len - length of a bin in ms (default 1s/39.0625 = 25.6ms)
            sampling_period - sampling period in ms (default 1/20kHz = 0.05ms)
            dtype - dtype to use for bins, default np.uint16 (np.uint8 would use less memory, but can store up to 256 spikes per bin)
        Return:
            Matrix (neuron num, time bins) of with spike count in each bin
    """
    st = spike_times_from_res_and_clu(res_path, clu_path)
    return bins_from_spike_times(st, bin_len, sampling_period)

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

# TODO unit tests
