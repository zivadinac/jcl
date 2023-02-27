import numpy as np
from jcl.load import bins_from_spike_times


def __bin_frs(bins, bin_ts, sampling_period, bl=10):
    ts_binned = bins_from_spike_times(bin_ts.T, bl, sampling_period).toarray()
    ts_b = np.where(ts_binned[0, :] > 0)[0]
    ts_e = np.where(ts_binned[1, :] > 0)[0]
    bin_d = (bin_ts[:, 1] - bin_ts[:, 0]) * sampling_period
    return [bins[:, b:e].sum(axis=1) / d for (b, e, d) in zip(ts_b, ts_e, bin_d)]


def classify_responses(spike_times, light_ts, sampling_period,
                       inh_thr=.75, disinh_thr=1.5):
    """ Classifiy cells based on firing rate change during light stimulation.
            'inh' - inhibited cells, FR <= `inh_thr` * baseline FR
            'disinh' - inhibited cells, FR >= `disinh_thr` * baseline FR
            'unresp' - neither of the previous two

        Args:
            spike_times - for the whole session, (list of lists)
            light_ts - timestamps of stimulation with light
            sampling_period - sampling period of data, need for FR calculation
            inh_thr - threshold for inhibited cells; default is .75
            disinh_thr - threshold for disinhibited cells; default is 1.5
    """
    BL = 10  # bin len in seconds for data binning
    light_ts_d = light_ts[:, 1] - light_ts[:, 0]  # durations
    light_ts_p = np.stack([light_ts[:, 0] - .5 * light_ts_d,
                           light_ts[:, 0]], axis=1)  # past
    bins = bins_from_spike_times(spike_times, BL, sampling_period).toarray()

    laser_frs = np.stack(__bin_frs(bins, light_ts, sampling_period, BL), axis=1)
    laser_frs = np.mean(laser_frs, axis=1)  # laser mean firing rates

    laser_r_frs = np.stack(__bin_frs(bins, light_ts_p, sampling_period, BL), axis=1)
    laser_r_frs = np.mean(laser_r_frs, axis=1)  # past mean firing rates

    assert laser_frs.shape == laser_r_frs.shape and\
           len(laser_frs) == len(spike_times)

    def __lr(laser_mean_fr, laser_p_mean_fr):
        if laser_mean_fr <= inh_thr * laser_p_mean_fr:
            return "inh"

        if laser_mean_fr >= disinh_thr * laser_p_mean_fr:
            return "disinh"

        return "unresp"

    return [__lr(l_fr, l_p_fr) for l_fr, l_p_fr in zip(laser_frs, laser_r_frs)]

