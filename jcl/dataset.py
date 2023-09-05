from abc import ABCMeta
import numpy as np
from os import listdir
from os.path import join, basename
from functools import cached_property
from jcl import load


_FULL_DAY_SN = "Full-day"


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, path, sampling_rate, whl_sampling_rate):
        self.__path = path
        self.__sampling_rate = sampling_rate
        self.__whl_sampling_rate = whl_sampling_rate

    @property
    def path(self):
        return self.__path

    @property
    def sampling_rate(self):
        return self.__sampling_rate

    @property
    def whl_sampling_rate(self):
        return self.__whl_sampling_rate

class RecordingDay(AbstractDataset):
    def __init__(self, path, sampling_rate, whl_sampling_rate):
        """ Contruct object for holding data from a single recording day.
            
            Args:
                path - path to the dataset (directory with clu, res, des... files)
                sampling_rate - sampling rate in Hz
                whl_sampling_rate - whl sampling rate in Hz
        """
        super().__init__(path, sampling_rate, whl_sampling_rate)
        self.__day = basename(path)
        self.__sessions = {_FULL_DAY_SN: [0]}
        self.__spike_times = {}
        self.__bins = {}
        self.__whl = {}
        self.__light = {}
        self.__swr = {}

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
            res_path = self.__make_path("res")
            clu_path = self.__make_path("clu")
            sls = self.session_limits
            sessions = self.get_session(sess_name)

            if len(sessions) == 1 or np.all(np.diff(sessions) == 1):
                fs, ls = sessions[0], sessions[-1]
                limits = sls[fs][0], sls[ls][1]
                sts = load.slice_spike_times(self.__spike_times[_FULL_DAY_SN], *limits)\
                      if _FULL_DAY_SN in self.__spike_times\
                      else spk_times(res_path, clu_path, exclude_clusters, limits)
                self.__spike_times[sess_name] = sts
            else:
                s_sts = [load.slice_spike_times(self.__spike_times[_FULL_DAY_SN], *sls[s])\
                         for s in sessions]\
                        if _FULL_DAY_SN in self.__spike_times\
                        else [spk_times(res_path, clu_path, exclude_clusters, sls[s])\
                              for s in sessions]
                n_cells = len(s_sts[0])
                assert np.all([len(st) == n_cells for st in s_sts])

                sts = [np.concatenate([st[i] for st in s_sts])
                       for i in range(n_cells)]
                self.__spike_times[sess_name] = sts
        return self.__spike_times[sess_name]

    def bins(self, bin_len, sess_name=_FULL_DAY_SN, exclude_clusters=[0, 1]):
        """ Load binned spike times for given sessions.

            Args:
                bin_len - bin duration in milliseconds
                sess_name - session name, default is Full-day
                exclude_clusters - clusters to exclude during loading (default is [0, 1])

            Return:
                bins - 2D array of shape (n_cells, n_bins)
        """
        try:
            return self.__bins[sess_name][bin_len]
        except KeyError:
            if self.__bins.get(sess_name) is None:
                self.__bins[sess_name] = {}
            clean_st = sess_name not in self.__spike_times
            st = self.spike_times(sess_name, exclude_clusters)
            sp = 1000 / self.sampling_rate
            self.__bins[sess_name][bin_len] = load.bins_from_spike_times(st, bin_len, sp)
            if clean_st:
                self.__spike_times.pop(sess_name)
        return self.__bins[sess_name][bin_len]

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
            whl = self.__whl[_FULL_DAY_SN] if _FULL_DAY_SN in self.__whl\
                  else load.positions_from_whl(self.__make_path("whl"))
            sls = self.session_limits
            sessions = self.get_session(sess_name)
            if len(sessions) == 1 or np.all(np.diff(sessions) == 1):
                fs, ls = sessions[0], sessions[-1]
                b = int(sls[fs][0] / self.sampling_rate * self.whl_sampling_rate)
                e = int(sls[ls][1] / self.sampling_rate * self.whl_sampling_rate)
                self.__whl[sess_name] = whl[b:e]
            else:
                s_whls = []
                for s in sessions:
                    b, e = sls[s]
                    b = int(b / self.sampling_rate * self.whl_sampling_rate)
                    e = int(e / self.sampling_rate * self.whl_sampling_rate)
                    s_whls.append(whl[b:e])
                self.__whl[sess_name] = np.vstack(s_whls)
        return self.__whl[sess_name]

    @cached_property
    def start_box(self):
        coords = load.readfromtxt(join(self.path, "start_box.txt"), int)
        return coords[0:2], coords[2:4]

    @cached_property
    def rewards(self):
        coords = load.readfromtxt(join(self.path, f"{self.day}.rewards"), int)
        assert len(coords) % 2 == 0
        return np.array(coords).reshape(-1, 2).tolist()

    def swr(self, sess_name=_FULL_DAY_SN):
        """ SWRs for given session.

            Args:
                sess_name - session name, default is Full-day

            Return:
                SWRs (array of shape (N, 3))
        """
        return self.__load_cols(sess_name, "sw", self.__swr)

    def light(self, sess_name=_FULL_DAY_SN):
        """ Timestamps of light stimulation.

            Args:
                sess_name - session name, default is Full-day

            Return:
                Array of shape (N, 2)
        """
        return self.__load_cols(sess_name, "light", self.__light)

    def __load_cols(self, sess_name, ext, holder):
        try:
            return holder[sess_name]
        except KeyError:
            cols = holder[_FULL_DAY_SN] if _FULL_DAY_SN in holder\
                   else np.array(load.sw(self.__make_path(ext)))
            sls = self.session_limits
            sessions = self.get_session(sess_name)

            if len(sessions) == 1 or np.all(np.diff(sessions) == 1):
                fs, ls = sessions[0], sessions[-1]
                b, e = sls[fs][0], sls[ls][1]
                idx = np.where((cols[:, 0] >= b) & (cols[:, -1] < e))[0]
                holder[sess_name] = cols[idx]
            else:
                s_cols = []
                for s in sessions:
                    b, e = sls[s]
                    idx = np.where((cols[:, 0] >= b) & (cols[:, -1] < e))[0]
                    s_cols.append(cols[idx])
                holder[sess_name] = np.vstack(s_cols)
        return holder[sess_name]

    def __make_path(self, ext):
        return join(self.path, f"{self.day}.{ext}")

class Dataset(AbstractDataset):
    def __init__(self, path, sampling_rate, whl_sampling_rate):
        """ Contruct dataset object.
            
            Args:
                path - path to the dataset (directory with a subdirectory for each day)
                sampling_rate - sampling rate in Hz
                whl_sampling_rate - whl sampling rate in Hz
        """
        super().__init__(path, sampling_rate, whl_sampling_rate)
        self.__days = {d: RecordingDay(join(path, d), sampling_rate, whl_sampling_rate)
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

