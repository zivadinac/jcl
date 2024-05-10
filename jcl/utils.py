import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline


def align_measured_data(population_vectors, positions, num_bins, speed=None):
    l = positions.shape[0] // num_bins
    if l < population_vectors.shape[1]:
        pv = population_vectors[:, 0:l]
        pos = positions[0:l*num_bins]
        s = speed[0:l*num_bins] if speed is not None else None
    else:
        ll = population_vectors.shape[1] * num_bins
        pv = population_vectors
        pos = positions[0:ll]
        s = speed[0:ll] if speed is not None else None

    sh = (-1, num_bins, 2) if pos.ndim == 2 else (-1, num_bins)
    pos = pos.reshape(sh).mean(axis=1)
    s = s.reshape(-1, num_bins).mean(axis=1) if speed is not None else None

    if s is not None:
        return pv, pos, s
    else:
        return pv, pos

def __interpolate_linear_position(position, unknown_val=-1, interp_order=1):
    """ Spline interpolation of missing values in animal position data.

        Args:
            position - coordinates
            unknown_val - placeholder for unknown/missing values
            interp_order - interpolation polynom degree, default 1 (linear)
        Return:
            Interpolated data in the same format as provided.
    """
    position = np.array(position)
    ok_inds = np.where(position != unknown_val)[0]
    f = InterpolatedUnivariateSpline(ok_inds, position[ok_inds], k=interp_order)
    return f(np.arange(len(position)))

def interpolate_position(x, y=None, unknown_val=-1, interp_order=1):
    """ Spline interpolation of missing values in animal position data.
        In case of 2d data (y not None), each dimension is interpolated independently.

        Args:
            x - coordinates
            y - coordinates (for 2d data only)
            unknown_val - placeholder for unknown/missing values
            interp_order - interpolation polynom degree, default 1 (linear)
        Return:
            Interpolated data in the same format as provided.
    """
    x_i = __interpolate_linear_position(x, unknown_val, interp_order)
    if y is not None:
        y_i = __interpolate_linear_position(y, unknown_val, interp_order)
        return x_i, y_i
    return x_i


def resample_whl(whl, new_len, whl_ts_col=None, whl_bl=None, interp_order=1):
    """ Spline interpolation of missing values in animal position data.

        Args:
            whl - position data
            new_len - len for resampled whl
            whl_ts_col - the column with timestamps; if None provide `whl_bl`
            whl_bl - whl bin len; used if whl doesn't have timestamps
            interp_order - interpolation polynom degree, default 1 (linear)
        Return:
            Interpolated data in the same format as provided.
    """
    if whl_ts_col is not None:
        orig_ts = whl[:, whl_ts_col]
    else:
        orig_ts = whl_bl * np.arange(len(whl))
    new_ts = orig_ts[-1] * np.arange(new_len) / new_len
    new_whl = -1 * np.ones((len(new_ts), whl.shape[1]))
    if whl_ts_col is not None:
        new_whl[:, -1] = new_ts
        last_c = whl.shape[1]-1
    else:
        last_c = whl.shape[1]
    for c in range(last_c):
        position = np.array(whl[:, c])
        position[position <= 0] = -1
        ok_idx = np.where(position != -1)[0]

        if position[0] == -1:
            position[0] = int(np.mean(position[ok_idx[:10]]))
        if position[-1] == -1:
            position[-1] = int(np.mean(position[ok_idx[-10:]]))

        position = interpolate_position(position)
        f = UnivariateSpline(orig_ts, position, k=interp_order)
        # prevent interpolation from going crazy
        p_l, p_h = position[ok_idx].min(), position[ok_idx].max()
        res_position = np.clip(f(new_ts), p_l, p_h)

        new_whl[:, c] = res_position
    return new_whl


def trial_duration(xy_trajectory, sampling_rate=39.0625):
    """ Compute trial duration in seconds based on given trajectory and sampling rate.

    Args:
        xy_trajectory - animal trajectory during a trial, array of shape (n,2)
        sampling_rate - number of samples per second (Hz), default 39.0625
    Return:
        Duration in seconds
    """
    return len(xy_trajectory) / sampling_rate  # in seconds


def trial_distance(xy_trajectory):
    """ Compute trial duration based on given trajectory.

    Args:
        xy_trajectory - animal trajectory during a trial, array of shape (n,2)
    Return:
        Distance covered by the trajectory (in the same units as xy_trajectory)
    """
    diffs = np.diff(xy_trajectory, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return distances.sum()


def calc_speed(positions, bin_len):
    """ Calculate animal speed. Units of positions / s.

        Args:
            positions - array of positions of shape (n,1) or (n,2)
            bin_len - duration of temporal bins in ms
        Return:
            speed - animal speed in each recorded point, same shape as positions
    """
    speed = []
    for i in range(1, len(positions)):
        prev_p = positions[i-1]
        current_p = positions[i]
        dist = np.linalg.norm(current_p-prev_p)
        speed.append(dist / (bin_len / 1000))
    return [speed[0]] + speed


def speed_filter(positions, bin_len, speed_thr):
    """ Filter positions based on speed.
        Return both parts where speed > speed_thr and where speed <= speed_thr.

            positions - array of positions of shape (n,1) or (n,2)
            bin_len - duration of temporal bins in ms
            speed_thr - threshold for speed
        Return:
            (high_speed_positions, low_speed_positions)
    """
    assert speed_thr is not None
    speed = np.array(calc_speed(positions, bin_len))
    speed_h_idx = speed > speed_thr
    speed_l_idx = speed <= speed_thr
    return positions[speed_h_idx], positions[speed_l_idx]


def get_last_spike_time(spike_times):
    """ Get the timestamp of the last spike.

        Args:
            spike_times - list of spike times per neuron (list of iterables, pre-sorted in a non-descending order)
        Return:
            Last spike timestamp (int)
    """
    return np.max([st[-1] for st in spike_times if len(st) > 0])


def get_first_spike_time(spike_times):
    """ Get the timestamp of the first spike.

        Args:
            spike_times - list of spike times per neuron (list of iterables, pre-sorted in a non-descending order)
        Return:
            First spike timestamp (int)
    """
    return np.min([st[0] for st in spike_times if len(st) > 0])


def compute_bins(spike_times, bin_len=25.6, sampling_period=0.05, limits=None):
    """ Compute bin edges for given spike times.

        Args:
            spike_times - list of spike times per neuron (list of iterables, pre-sorted in a non-descending order)
            bin_len - length of a bin in ms (default 1s/39.0625 = 25.6ms)
            sampling_period - sampling period in ms (default 1s/20kHz = 0.05ms)
            limits - session limits (begin, end) in ms, if None use first and last spike time
        Return:
            List of bin edges (in ms), total number of bins
    """
    if limits is None:
        last_spike_time = to_ms(get_last_spike_time(spike_times), sampling_period)
        first_spike_time = to_ms(get_first_spike_time(spike_times), sampling_period)
        limits = (first_spike_time, last_spike_time)
    bin_num = np.ceil(np.round((limits[1] - limits[0]) / bin_len, 3)).astype(int)
    bin_edges = limits[0] + np.arange(bin_num + 1) * bin_len
    return bin_edges, bin_num


def compute_constant_bins(spike_num, spike_times=None, res=None, sampling_period=0.05):
    """ Compute bin edges such that every bin contains `spike_num` spikes.

        Args:
            spike_num - number of spikes per bin
            spike_times - list of spike times per neuron (list of iterables, pre-sorted in a non-descending order)
            res - spike times; used only if `spike_times` is None
            sampling_period - sampling period in ms (default 1s/20kHz = 0.05ms)
        Return:
            List of bin edges (in ms), total number of bins
    """
    assert sum([res is not None, spike_times is not None]) == 1,\
           "Provide only `spike_times` or only `res`"
    if res is None:
        res = []
        for st in spike_times:
            res.extend(st)
        res = sorted(res)
    res = to_ms(np.array(res), sampling_period)
    rem = len(res) % spike_num
    if rem > 0:
        resm = res[:-rem]
    else:
        resm = res
    resm = resm.reshape(-1, spike_num)
    b, e = res[0], res[-1]
    lims = [b] + resm[:, -1].tolist()
    if rem > 0:
        lims = lims + [e]
    return lims, len(lims)


def concatenate_spike_times(*all_spike_times, shift_sessions=True, session_lims=None):
    """ Concatenate spike times from different sessions (shift appropriately).

        Args:
            all_spike_times - spike times (list of lists) for every session
            shift_sessions - if True (default) shift spike times for every session in `all_spike_times`
            session_lims - session limits - list of (begin_ts, end_ts),
                           should be of the same length as `all_spike_times`,
                           if None (default) first and last spike time are used
        Return:
            concatenated spike times (list of lists)
    """

    unit_nums = np.array([len(st) for st in all_spike_times])
    if not np.all(unit_nums == unit_nums[0]):
        raise ValueError("All given spike times (sessions) must have equal number of units.")
    if session_lims is not None:
        assert len(session_lims) == len(all_spike_times)
        last_spike_times = [sl[1] for sl in session_lims]
    else:
        last_spike_times = [get_last_spike_time(st) for st in all_spike_times]
    if shift_sessions:
        session_shifts = [0] + np.cumsum(last_spike_times)[:-1].tolist()
    else:
        session_shifts = [0 for i in range(len(all_spike_times))]
    assert len(session_shifts) == len(all_spike_times)

    cat_spike_times = []

    for u in range(unit_nums[0]):
        all_st_u = [np.array(st[u]) for st in all_spike_times]
        all_st_u_shifted = [all_st_u[i] + session_shifts[i] for i in range(len(all_st_u))]
        cat_spike_times.append(np.concatenate(all_st_u_shifted))

    return cat_spike_times

def to_ms(x, sampling_period):
    """ Convert time `x` to ms.

        Args:
            x - time to be converted (number or array of numbers)
            sampling_period - sampling period in ms
        Return:
            Converted data of the same type as `x`
    """
    return x * sampling_period

def __elapsed(st, sampling_period):
    """ Compute elapsed time (in s) between the first and the last spike in given list of spike times."""
    return None if len(st) <= 0 else sampling_period * (st[-1]-st[0]) / 1000

def compute_frs(spike_times, sampling_period=0.05):
    """ Compute firing rate (in Hz) for each neuron in given spike times.

        Args:
            spike_times - list of spike times per neuron (list of iterables, pre-sorted in a non-descending order)
            sampling_period - sampling period in ms (default 1s/20kHz = 0.05ms)
        Return:
            List of firing rates.
    """
    return [len(st) / __elapsed(st, sampling_period) if len(st) > 0 else 0 for st in spike_times]

def generate_sliding_windows(bins, window_len, stride=None):
    """ Generate sliding windows over given bins.

        Args:
            bins - matrix with binned data
            window_len - length of each window in number of bins
            stride - stride between windows (defaults to window_len)
        Return:
            Generator with slices of bins.
    """
    if stride is None:
        stride = window_len
    for start in range(0, bins.shape[1], stride):
        yield bins[:, start:start+window_len]

def traj_in_interval_1D(traj, interval):
    """ Check which points of the trajectory are in the given interval.

        Args:
            traj - trajectory, array with `n` elements
            interval - (begin, end), inclusive
        Return:
            True / False array with `n` elements
    """
    return (traj >= interval[0]) & (traj <= interval[1])

def traj_in_rect_2D(traj, rect):
    """ Check which points of the trajectory are in the given rect.

        Args:
            traj - trajectory, array of shape (n, 2)
            rect - coordinates of the rect [(TL_x, TL_y), (BR_x, BR_y)]

        Return:
            True / False array with `n` elements
    """
    (TL_x, TL_y), (BR_x, BR_y) = rect
    x_idx = traj_in_interval_1D(traj[:, 0], (TL_x, BR_x))
    y_idx = traj_in_interval_1D(traj[:, 1], (BR_y, TL_y))
    return np.logical_and(x_idx, y_idx)

def traj_in_circle_2D(traj, center, radius):
    """ Check which points of the trajectory are in the given circle.

        Args:
            traj - trajectory, array of shape (n, 2)
            center - coordinates of the center (c_x, c_y)
            radius - circle radius

        Return:
            True / False array with `n` elements
    """
    return np.linalg.norm(traj - np.array(center), axis=1) ** 2 <= radius ** 2

def res_clu_from_spike_times(st):
    """ Generate res and clu for given spike times.

        Args:
            st - spike times, list of lists

        Return:
            res, clu - np.ndarrays
    """
    clu = np.concatenate([len(st_c) * [c] for c, st_c in enumerate(st)])
    res = np.concatenate(st).astype(np.int64)
    sort_idx = np.argsort(res)
    return res[sort_idx], clu[sort_idx]

def autocorrelogram(spike_times, dur=51, n_spk=100, sampling_rate=20000):
    """ Compute spike train autocorrelogram.

        Args:
            spike_times - list of spike times
            dur - histogram duration in ms (default is 50)
            n_spk - number of spikes in future and past to use for histogram
                    (default is 100)
            sampling_rate - sampling rate in Hz
        Return:
            (times, counts)
    """
    bin_edges = np.arange(-dur / 2, dur / 2 + 1)  # ms
    sts = to_ms(spike_times, 1000 / sampling_rate)
    cum_hist = np.zeros(len(bin_edges) - 1)
    for i, st in enumerate(sts):
        s = np.maximum(0, i-n_spk)
        dst = st - sts[s:i+n_spk+1]
        dst[i-s] = np.nan
        cum_hist += np.histogram(dst, bin_edges)[0]
    plot_edges = (bin_edges[1:] + bin_edges[:-1]) / 2
    return plot_edges, cum_hist

# TODO unit tests


"""
# quick test

x = [1, 2, -1, 4, 5, 6]
y = [-1, 2, 3, -1, 5, -1]

x_i, y_i = interpolate_position(x, y)


#pv, pp, ss = align_measured_data(fr, p, 4, speed=s)
"""
