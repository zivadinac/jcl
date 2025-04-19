from collections import namedtuple
from dataclasses import dataclass, field
import numpy as np
from os import listdir
from os.path import join, basename
from functools import cached_property
from jcl import load
from jcl.utils import concatenate_spike_times


_FULL_DAY_SN = "Full-day"


@dataclass
class AbstractDataset():
    path: str
    n_tetrodes: int
    sampling_rate: float
    whl_sampling_rate: float
    eeg_sampling_rate: float
    eegh_sampling_rate: float

    @property
    def n_channels(self):
        return self.n_tetrodes * 4

class RecordingDay(AbstractDataset):
    def __init__(self, path, n_tetrodes,
                 sampling_rate=20000,
                 whl_sampling_rate=39.0625,
                 eeg_sampling_rate=1250,
                 eegh_sampling_rate=5000):
        """ Contruct object for holding data from a single recording day.

            Args:
                path - path to the dataset (directory with clu, res, des... files)
                sampling_rate - sampling rate in Hz
                whl_sampling_rate - whl sampling rate in Hz
        """
        super().__init__(path, n_tetrodes,
                         sampling_rate, whl_sampling_rate,
                         eeg_sampling_rate, eegh_sampling_rate)
        self.__day = basename(path)
        self.__sessions = {_FULL_DAY_SN: [0]}
        self.__spike_times = {}
        self.__bins = {}
        self.__bin_edges = {}
        self.__whl = {}
        self.__light = {}
        self.__rem = {}
        self.__nrem = {}
        self.__sleep = {}
        self.__awake = {}
        self.__awake_rest = {}
        self.__swr = {}
        self.__hse = {}
        self.__st_limits = {}
        self.__eeg = {}
        self.__eegh = {}

    def clear(self, clear_sessions=False):
        """ Delete all data loaded by this instance.

            Args:
                clear_sessions - clear defined sessions (default False)
        """
        def __clear_dict(dic):
            for v in dic.values():
                if type(v) == dict:
                    __clear_dict(v)
                else:
                    del v
            dic.clear()

        __clear_dict(self.__spike_times)
        __clear_dict(self.__bins)
        __clear_dict(self.__bin_edges)
        __clear_dict(self.__whl)
        __clear_dict(self.__light)
        __clear_dict(self.__rem)
        __clear_dict(self.__nrem)
        __clear_dict(self.__sleep)
        __clear_dict(self.__awake)
        __clear_dict(self.__awake_rest)
        __clear_dict(self.__swr)
        __clear_dict(self.__hse)
        __clear_dict(self.__st_limits)
        __clear_dict(self.__eeg)
        __clear_dict(self.__eegh)
        if clear_sessions:
            self.__sessions.clear()

    @property
    def day(self):
        return self.__day

    @property
    def sessions(self):
        """ Return dictionary mapping session names to recorgin session numbers. """
        return self.__sessions

    def get_session(self, sess_name):
        """ Get list of recording session numbers for given session name."""
        try:
            return self.__sessions[sess_name]
        except KeyError:
            raise ValueError(f"Undefined session {sess_name}. Use .set_session() method to define it.")

    def get_session_limits(self, sess_name):
        """ Get list of recording session numbers for given session name."""
        sl = self.get_session(sess_name)
        if len(sl) == 1:
            return self.session_limits[sl[0]]
        else:
            return self.session_limits[sl[0]][0], self.session_limits[sl[-1]][1]

    def get_session_duration_s(self, sess_name="Full-day"):
        dur = 0
        for sn in self.get_session(sess_name):
            sl = self.session_limits[sn]
            dur += (sl[1]-sl[0]) / self.sampling_rate
        return dur

    def set_session(self, sess_name, sess_nums):
        """ Define session `sess_name` composed of recording sessions indicated by `sess_nums`.

            Args:
                sess_name - name of the session
                sess_nums - list of recording sessions in ascending order
        """
        if sess_name in self.__sessions:
            raise ValueError(f"Session {sess_name} already defined.")
        if np.any(np.diff(sess_nums) <= 0):
            raise ValueError(f"Provide `sess_nums` in ascending order.")
        if np.min(sess_nums) <= 0:
            raise ValueError(f"`sess_nums` must contain only positive numbers.")
        self.__sessions[sess_name] = sess_nums

    def set_sessions_from_desen(self):
        """ Defines sessions based on the first column of .desen file. """
        sns = np.array([row.split(' ')[0].strip() for row in self.desen])
        for sn in np.unique(sns):
            sn_nums = np.where(sns == sn)[0] + 1
            self.set_session(sn, sn_nums.tolist())

    @cached_property
    def info(self):
        """ Read .info file - text file with information about the recording day.
            Format of the file (all coordinates in pixels):

            pixels_per_cm (float)
            starting_box_TL_x, starting_box_TL_y, starting_box_BR_x, starting_box_BR_y
            maze_center_x, maze_center_y
            reward_new_1_x, reward_new_1_y, reward_new_2_x, reward_new_2_y, reward_new_3_x, reward_new_3_y
            reward_old_1_x, reward_old_1_y, reward_old_2_x, reward_old_2_y, reward_old_3_x, reward_old_3_y
        """
        with open(self.__make_path("info"), "r") as f:
            lines = f.readlines()
            try:
                pix_per_cm = float(lines[0].strip())
            except Exception:
                pix_per_cm = None
            try:
                SB = lines[1].strip().split(' ')
                assert len(SB) == 4
                SB = [(float(SB[0]), float(SB[1])),
                      (float(SB[2]), float(SB[3]))]
            except Exception:
                SB = None
            try:
                maze_center = (float(c) for c in lines[2].strip().split(' '))
            except Exception:
                maze_center = None
            try:
                new_r = lines[3].strip().split(' ')
                assert len(new_r) % 2 == 0
                rewards_new = [(float(new_r[2*i]), float(new_r[2*i+1]))
                               for i in range(len(new_r) // 2)]
            except Exception:
                rewards_new = None
            try:
                old_r = lines[4].strip().split(' ')
                assert len(old_r) % 2 == 0
                rewards_old = [(float(old_r[2*i]), float(old_r[2*i+1]))
                               for i in range(len(old_r) // 2)]
            except Exception:
                rewards_old = None
            Info = namedtuple("Info", "pix_per_cm SB maze_center rewards_new rewards_old")
            info = Info(pix_per_cm, SB, maze_center, rewards_new, rewards_old)
        return info

    @cached_property
    def resofs(self):
        return load.readfromtxt(self.__make_path("resofs"), int)

    @cached_property
    def session_limits(self):
        """ Session limits.

            List with n_sessions + 1 elements.
            Element at index 0 holds limits for the whole recording day.
            Each element is a tuple holding beginning and end timestamp (b_ts, e_ts).
        """
        sls = load.session_limits(self.__make_path("resofs"))
        return [(0, sls[-1][1])] + sls

    @cached_property
    def desen(self):
        return load.readfromtxt(self.__make_path("desen"), lambda s: s.strip())

    @cached_property
    def des(self):
        return load.readfromtxt(self.__make_path("des"), lambda s: s.strip())

    @cached_property
    def num_cells(self):
        return len(self.des) - 1

    def spike_times(self, sess_name=_FULL_DAY_SN, exclude_clusters=[0, 1]):
        """ Load spike times for given session.

            Args:
                sess_name - session name, default is Full-day
                exclude_clusters - clusters to exclude during loading (default is [0, 1])

            Return:
                Spike times (list of np.arrays)
        """

        try:
            return self.__spike_times[sess_name]
        except KeyError:
            spk_times = load.spike_times_from_res_and_clu
            sls = self.session_limits
            sessions = self.get_session(sess_name)

            if len(sessions) == 1:
                sn = None if sess_name == _FULL_DAY_SN else sessions[0]
                res_path = self.__make_path("res", sn)
                clu_path = self.__make_path("clu", sn)
                fs, ls = sessions[0], sessions[-1]
                limits = sls[fs][0], sls[ls][1]
                sts = spk_times(res_path, clu_path, exclude_clusters)
                self.__spike_times[sess_name] = sts
                self.__st_limits[sess_name] = limits
            else:
                s_sts = []
                for sn in sessions:
                    res_path = self.__make_path("res", sn)
                    clu_path = self.__make_path("clu", sn)
                    s_sts.append(spk_times(res_path, clu_path, exclude_clusters))
                n_cells = len(s_sts[0])
                assert np.all([len(st) == n_cells for st in s_sts])
                s_lims = [self.session_limits[s] for s in sessions]
                ss = [0] + np.cumsum([sl[1]-sl[0] for sl in s_lims]).tolist()
                sts = [np.concatenate([np.array(st[i]) + ss[sn]
                       for sn, st in enumerate(s_sts)])
                       for i in range(n_cells)]
                self.__spike_times[sess_name] = sts
                self.__st_limits[sess_name] = (ss[0], ss[-1])
        return self.__spike_times[sess_name]

    def bins(self, bin_len, sess_name=_FULL_DAY_SN, exclude_clusters=[0, 1], binning_fun=None):
        """ Load binned spike times for given sessions.

            Args:
                bin_len - bin duration in milliseconds
                sess_name - session name, default is Full-day
                exclude_clusters - clusters to exclude during loading (default is [0, 1])
                binning_fun - callable that computes bins based on spike times, default is None - use fixed `bin_len`
            Return:
                bins - 2D array of shape (n_cells, n_bins)
        """
        try:
            return self.__bins[sess_name][bin_len]
        except KeyError:
            if self.__bins.get(sess_name) is None:
                self.__bins[sess_name] = {}
                self.__bin_edges[sess_name] = {}
            clean_st = sess_name not in self.__spike_times
            st = self.spike_times(sess_name, exclude_clusters)
            sp = 1000 / self.sampling_rate
            limits = self.__st_limits[sess_name]
            limits = (0, limits[1]-limits[0])
            bins = binning_fun(st) if binning_fun else bin_len
            bs, be = load.bins_from_spike_times(st, bins, sp, limits=limits,
                                                return_edges=True)
            self.__bins[sess_name][bin_len] = bs
            self.__bin_edges[sess_name][bin_len] = be
            if clean_st:
                self.__spike_times.pop(sess_name)
        return self.__bins[sess_name][bin_len]

    def bin_edges(self, bin_len, sess_name):
        """ Return bin edges for given length and session.
            Bins must be precomputed in order to have bin edges. """
        return self.__bin_edges[sess_name][bin_len]

    def whl(self, sess_name=_FULL_DAY_SN):
        """ Raw whl for given session.

            Args:
                sess_name - session name, default is Full-day

            Return:
                whl (array of shape (N, ...))
        """
        try:
            return self.__whl[sess_name]
        except KeyError:
            sessions = self.get_session(sess_name)
            s_whls = []
            for s in sessions:
                whl = load.positions_from_whl(self.__make_path("whl", s))
                s_whls.append(whl)
            self.__whl[sess_name] = np.vstack(s_whls)
        return self.__whl[sess_name]

    def eegh(self, sess_name=_FULL_DAY_SN):
        """ Load eegh for given session.

            Args:
                sess_name - session name, default is Full-day

            Return:
                eegh (array of shape (num_sample, num_tetrodes)
        """
        return self.__binary(sess_name, self.__eegh, "eegh",
                             self.eegh_sampling_rate, self.n_tetrodes)

    def eeg(self, sess_name=_FULL_DAY_SN):
        """ Load eeg for given session.

            Args:
                sess_name - session name, default is Full-day

            Return:
                eeg (array of shape (num_sample, num_channels)
        """
        return self.__binary(sess_name, self.__eeg, "eeg",
                             self.eeg_sampling_rate, self.n_channels)

    def __load_binary(self, path, n_ch):
        return np.fromfile(path, dtype=np.int16).reshape(-1, n_ch)

    def __binary(self, sess_name, holder, ext, sr, n_ch):
        try:
            return holder[sess_name]
        except KeyError:
            sessions = self.get_session(sess_name)
            # load from disk
            # assume that each session has a separate .eegh file
            paths = [join(self.path, f"{self.day}_{sn}.{ext}")
                     for sn in sessions]
            eeghs = [self.__load_binary(p, n_ch)
                     for p in paths]
            holder[sess_name] = np.concatenate(eeghs)
            return holder[sess_name]

    @cached_property
    def start_box(self):
        coords = load.readfromtxt(join(self.path, "start_box.txt"), int)
        return coords[0:2], coords[2:4]

    @cached_property
    def rewards(self):
        coords = load.readfromtxt(join(self.path, f"{self.day}.rewards"), int)
        assert len(coords) % 2 == 0
        return np.array(coords).reshape(-1, 2).tolist()

    def hse(self, ct, sess_name=_FULL_DAY_SN):
        """ SWRs for given session.

            Args:
                sess_name - session name, default is Full-day

            Return:
                SWRs (array of shape (N, 3))
        """
        if ct not in self.__hse.keys():
            self.__hse[ct] = {}
        return self.__load_cols_2(sess_name, f"hse.{ct}", self.__hse[ct])

    def swr(self, sess_name=_FULL_DAY_SN):
        """ SWRs for given session.

            Args:
                sess_name - session name, default is Full-day

            Return:
                SWRs (array of shape (N, 3))
        """
        return self.__load_cols_2(sess_name, "sw", self.__swr)

    def light(self, sess_name=_FULL_DAY_SN):
        """ Timestamps of light stimulation.

            Args:
                sess_name - session name, default is Full-day

            Return:
                Array of shape (N, 2)
        """
        return self.__load_cols_2(sess_name, "light", self.__light)

    def rem(self, sess_name=_FULL_DAY_SN):
        """ Timestamps of REM sleep.

            Args:
                sess_name - session name, default is Full-day

            Return:
                Array of shape (N, 2)
        """
        return self.__load_cols_2(sess_name, "srem", self.__rem)

    def nrem(self, sess_name=_FULL_DAY_SN):
        """ Timestamps of NREM sleep.

            Args:
                sess_name - session name, default is Full-day

            Return:
                Array of shape (N, 2)
        """
        return self.__load_cols_2(sess_name, "snrem", self.__nrem)

    def slp(self, sess_name=_FULL_DAY_SN):
        """ Timestamps of sleep.

            Args:
                sess_name - session name, default is Full-day

            Return:
                Array of shape (N, 2)
        """
        return self.__load_cols_2(sess_name, "slp", self.__sleep)

    def nslp(self, sess_name=_FULL_DAY_SN):
        """ Timestamps of sleep.

            Args:
                sess_name - session name, default is Full-day

            Return:
                Array of shape (N, 2)
        """
        return self.__load_cols_2(sess_name, "nslp", self.__awake)

    def awake_rest(self, sess_name=_FULL_DAY_SN):
        """ Timestamps of awake rest epochs.

            Args:
                sess_name - session name, default is Full-day

            Return:
                Array of shape (N, 2)
        """
        return self.__load_cols_2(sess_name, "awr", self.__awake_rest)

    def __load_cols_2(self, sess_name, ext, holder):
        try:
            return holder[sess_name]
        except KeyError:
            if sess_name == _FULL_DAY_SN:
                # all sessions together, but load them one at a time
                sessions = list(range(1, len(self.session_limits)))
            else:
                sessions = self.get_session(sess_name)
            sls = self.session_limits
            sld = [sls[s][1]-sls[s][0] for s in sessions]
            offsets = [0] + np.cumsum(sld[:-1]).tolist()
            s_cols = []
            for s, o in zip(sessions, offsets):
                p = join(self.path, f"{self.day}_{s}.{ext}")
                c = np.array(load.sw(p))
                if c.size > 0:
                    s_cols.append(c + o)
            if len(s_cols) == 0:
                s_cols.append([])
            holder[sess_name] = np.vstack(s_cols)
            return holder[sess_name]

    def __make_path(self, ext, sess_num=None):
        if sess_num:
            return join(self.path, f"{self.day}_{sess_num}.{ext}")
        else:
            return join(self.path, f"{self.day}.{ext}")

class Dataset(AbstractDataset):
    def __init__(self, path, n_tetrodes,
                 sampling_rate=20000,
                 whl_sampling_rate=39.0625,
                 eeg_sampling_rate=1250,
                 eegh_sampling_rate=5000):
        """ Contruct dataset object.

            Args:
                path - path to the dataset (directory with a subdirectory for each day)
                n_tetrodes - number of tetrodes
                sampling_rate - sampling rate in Hz (default is 20_000)
                whl_sampling_rate - whl sampling rate in Hz (default is 39.0625)
                eeg_sampling_rate - whl sampling rate in Hz (default is 1250)
                eegh_sampling_rate - whl sampling rate in Hz (default is 5000)
        """
        super().__init__(path, n_tetrodes,
                         sampling_rate, whl_sampling_rate,
                         eeg_sampling_rate, eegh_sampling_rate)
        self.__days = {d: RecordingDay(join(path, d), n_tetrodes,
                                       sampling_rate, whl_sampling_rate,
                                       eeg_sampling_rate, eegh_sampling_rate)
                       for d in self._list_days(self.path)}

    @property
    def days(self):
        return self.__days

    def __getitem__(self, day):
        return self.days[day]

    @staticmethod
    def _list_days(path):
        return [d for d in listdir(path)
                if d.startswith("jc") or
                d.startswith("mjc") or
                d.startswith("mDRCCK")]

def combine_datasets(datasets):
    ds1 = datasets[0]
    ds = AbstractDataset("", ds1.n_tetrodes,
                         ds1.sampling_rate, ds1.whl_sampling_rate,
                         ds1.eeg_sampling_rate, ds1.eegh_sampling_rate)
    ds.days = {}
    for dss in datasets:
        ds.days = ds.days | dss.days
    return ds


"""
from time import time

day = "mjc169R4R_0114"
sn = "Sleep-long"

ds = Dataset("raw_data/peter", 20_000, 39.0625)
ds[day].set_session(sn, [6, 7, 8, 9, 10])

swr_whole = ds[day].swr()
whl_whole = ds[day].whl()
b = time()
swr_ls = ds[day].swr(sess_name="Sleep-long")
whl_ls = ds[day].whl("Sleep-long")
e = time()
print("swr + whl sn: ", e-b)
#bins_whole_51_2 = ds[day].bins(day, 51.2)

b = time()
st_whole = ds[day].spike_times()
e = time()
print("spike times full: ", e-b)

b = time()
bins_ls = ds[day].bins(102.4, "Sleep-long")
e = time()
print("bins sn: ", e-b)

b = time()
bins_ls = ds[day].bins(102.4, "Sleep-long")
e = time()
print("bins sn", e-b)

b = time()
bins_whole_102_4 = ds[day].bins(102.4)
e = time()
print("bins whole:", e-b)

b = time()
st_ls = ds[day].spike_times(sess_name="Sleep-long") # ~11s
e = time()
print("spike times sn:", e-b)

ds[day].set_session("Sleep-long-be", [6, 10])
b = time()
st_ls_be = ds[day].spike_times(sess_name="Sleep-long-be") # ~19s
e = time()
print("spike times sn_b:", e-b)
"""

