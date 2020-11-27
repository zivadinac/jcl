import numpy as np
from os.path import join

def spike_times_from_res_and_clu(res_path, clu_path):
    """ Load spike times for each neuron from provided '.res' and '.clu' files.
        
        Args:
            res_path - path to res file
            clu_path - path to clu file
        Return:
            List of firing times for each cell (list of lists)
    """
    clu = np.genfromtxt(clu_path, dtype=int)
    res = np.genfromtxt(res_path, dtype=int)
    assert len(clu) == len(res) or len(clu) == len(res) + 1

    if len(clu) == len(res) + 1: # has number of clusters written in the first line
        clu = clu[1:]
                                               
    spike_times = []
    for i in range(2, np.max(clu) + 1): # cluster 0 and 1 are noise
        spike_times.append(res[np.where(clu == i)[0]])
    return spike_times

def population_vectors_from_spike_times(spike_times, bin_len=102.4, bin_overlap=0., sampling_period=0.05):
    """ Produce population vectors for given spike times.

        Args:
            spike_times - list of spike time for each neuron (list of lists)
            bin_len - length of a bin in ms
            bin_overlap - temporal overlap between succesive bins
            sampling_period - sampling period in ms (default 1/20kHz = 0.05ms)
        Return:
            Matrix (neuron num, time bins) of with spike count in each bin
    """
    # ms, for 20kHz sampling rate
    maxes = [np.max(st) if len(st) > 0 else 0 for st in spike_times]
    last_spike_time = sampling_period * np.max(maxes) # in ms
    bin_num = np.ceil(last_spike_time / bin_len).astype(int)
    if bin_overlap > 0.:
        bin_num += (bin_num - 1)
    population_vectors = np.zeros((len(spike_times), bin_num))

    for n, st in enumerate(spike_times):
        st_ms = np.array(st) * sampling_period
        #odd_bin_inds = ((np.array(st) * sampling_period) // bin_len).astype(int)
        #even_bin_inds = ((np.array(st[st > bin_overlap]) * sampling_period) // bin_len).astype(int)
        for i in range(bin_num):
            l = i * bin_overlap if bin_overlap > 0 else i * bin_len
            h = l + bin_len
            spikes_in_bin_i = np.logical_and(st_ms >= l, st_ms < h)
            population_vectors[n, i] += len(st_ms[spikes_in_bin_i])
    return population_vectors

def population_vectors(res_path, clu_path, bin_len=102.4, bin_overlap=0., sampling_period=0.05):
    """ Produce population vectors for given '.res' and '.clu' files.

        Args:
            res_path - path to res file
            clu_path - path to clu file
            bin_len - length of a bin in ms
            bin_overlap - temporal overlap between succesive bins
            sampling_period - sampling period in ms (default 1/20kHz = 0.05ms)
        Return:
            Matrix (neuron num, time bins) of with spike count in each bin
    """
    st = spike_times_from_res_and_clu(res_path, clu_path)
    return population_vectors_from_spike_times(st, bin_len, bin_overlap, sampling_period)

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
    return d[:, [0,1]], d[:, 2]

"""
# quick test
# TODO unit tests
st = [[4,7,12,17,23,27,35,41,43,48]]
pv = population_vectors_from_spike_times(st, bin_len=10, bin_overlap=5, sampling_period=1)
data_folder = "/home/pzivadin/external_drive/Dama_merged/mDRCCK17-30052018-0110"
basename = join(data_folder, "mDRCCK17-30052018-0110_2")
spike_times = spike_times_from_res_and_clu(basename + ".res", basename + ".clu")
fr = population_vectors_from_spike_times(spike_times, bin_len=307.2, bin_overlap=153.6)
fr_2 = population_vectors_from_spike_times(spike_times, bin_len=307.2)
p,s = positions_and_speed_from_whl2(basename + ".whl2")
ct = cell_types(basename + ".des")
"""
