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

def population_vectors_from_spike_times(spike_times, bin_len=102.4, sampling_period=0.05):
    """ Produce population vectors for given spike times.

        Args:
            spike_times - list of spike time for each neuron (list of lists) bin_len - length of a bin in ms
            sampling_period - sampling period in ms (default 1/20kHz = 0.05ms)
        Return:
            Matrix (neuron num, time bins) of with spike count in each bin
    """
    # ms, for 20kHz sampling rate
    maxes = [np.max(st) if len(st) > 0 else 0 for st in spike_times]
    last_spike_time = sampling_period * np.max(maxes) # in ms
    bin_num = np.ceil(last_spike_time / bin_len).astype(int)
    population_vectors = np.zeros((len(spike_times), bin_num+1))

    for n, st in enumerate(spike_times):
        bin_inds = ((np.array(st) * sampling_period) // bin_len).astype(int)
        for i in bin_inds:
            population_vectors[n, i] += 1
    return population_vectors

def population_vectors(res_path, clu_path, bin_len=100, sampling_period=0.05):
    """ Produce population vectors for given '.res' and '.clu' files.

        Args:
            res_path - path to res file
            clu_path - path to clu file
            bin_len - length of a bin in ms
            sampling_period - sampling period in ms (default 1/20kHz = 0.05ms)
        Return:
            Matrix (neuron num, time bins) of with spike count in each bin
    """
    st = spike_times_from_res_and_clu(res_path, clu_path)
    return population_vectors_from_spike_times(st, bin_len, sampling_period)

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
data_folder = "/home/pzivadin/external_drive/Dama_merged/mDRCCK17-30052018-0110"
basename = join(data_folder, "mDRCCK17-30052018-0110_2")
spike_times = spike_times_from_res_and_clu(basename + ".res", basename + ".clu")
fr = population_vectors_from_spike_times(spike_times)
#fr_2 = population_vectors_from_spike_times(spike_times, bin_len=0.05)
p,s = positions_and_speed_from_whl2(basename + ".whl2")
ct = cell_types(basename + ".des")
"""
