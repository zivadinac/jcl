from functools import cached_property
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cosine
from scipy.signal import convolve2d


class Binning:
    def __init__(self, bin_size, maze_size):
        """ Calculate 2D bins.

            Args:
                bin_size - in the same units as positions for which bins are calculated
                maze_size - tuple, in the same units as bin_size;
        """
        self.bin_size = bin_size
        self.maze_size = maze_size

    def bin_idx(self, p):
        """ Calculate bin index for given position.

            Args:
                p - position
                bin_size - size of bins, single number or same shape as `p`
                maze_size - size of the maze (number of bins per dimension)
            Return:
                bin index - tuple of the same shape as `p`
        """
        return tuple((p // self.bin_size).astype(int))

    @cached_property
    def num_bins(self):
        """ Calculate number of spatial bins (per dimension). """
        return 1 + (np.array(self.maze_size) / self.bin_size).astype(int)


class Map:
    def __init__(self, mmap):
        self.__map = mmap
        self.__map_prob = None

    @property
    def map(self):
        """ Return the underlying map. """
        return self.__map

    @property
    def map_prob(self):
        """ Return the underlying map normalized to sum of 1. """
        if self.__map_prob is None:
            self.__map_prob = self.__map / self.__map.sum()
        return self.__map_prob

    # a map is an array
    # so we implement methods that are
    # absolutely essential for an array
    def __getitem__(self, idx):
        return self.__map.__getitem__(idx)

    @property
    def ndim(self):
        return self.__map.ndim

    @property
    def shape(self):
        return self.__map.shape

    @property
    def size(self):
        return self.__map.size

    def shift(self, shift, axis=None):
        """ Shift map along given axis.
            Args:
                shift - amount to shift by
                axis - axis to shift along
                       if None (default) shift along all axes
            Return:
                shifted Map
        """
        shifted_map = self.map.copy()
        axes = range(self.map.ndim) if axis is None else [axis]
        for a in axes:
            shifted_map = np.roll(shifted_map, shift, axis=a)
        return type(self)(mmap=shifted_map)

    def random_shift(self, axis=None):
        """ Randomly shift map along given axis.
            The amount of shift is at most map.shape-1.
            Args:
                axis - axis to shift along
                       if None (default) shift randomly along all axes
            Return:
                randomly shifted Map
        """
        shifted_map = self.map.copy()
        axes = range(self.map.ndim) if axis is None else [axis]
        shifts = [np.random.randint(1, self.shape[a] - 1) for a in axes]
        for a, s in zip(axes, shifts):
            shifted_map = np.roll(shifted_map, s, axis=a)
        return type(self)(mmap=shifted_map)



class OccupancyMap(Map):
    def __init__(self, positions, maze_size, bin_size, bin_len, min_thr=.256, smooth_sd=None, mmap=None):
        """ Produce occupancy map.

            Args:
                positions - array of positions of shape (n,1) or (n,2)
                maze_size - size of the maze in the same units as positions
                bin_size - size of spatial bins in the same units as positions
                bin_len - duration of temporal bins in ms
                min_thr - zero-out bins with occupancy smaller than this
                smooth_sd - SD in bins for gaussian smoothing
                mmap - map to use as occupancy map, if provided previous args should be None
            Return:
                occupancy - time in seconds spent in each spatial bin
        """
        if mmap is not None:
            occ = mmap
            bin_size = None
            binner = None
        else:
            if maze_size is None:
                maze_size = np.ceil(positions.max(axis=0)).astype(int) + 1
            binner = Binning(bin_size, maze_size)
            occ = self.__compute_occ(positions, binner, bin_len, min_thr, smooth_sd)
        super().__init__(occ)
        self.bin_size = bin_size
        self.binner = binner

    @staticmethod
    def __compute_occ(positions, binner, bin_len, min_thr, smooth_sd=None):
        occupancy = np.zeros(binner.num_bins)

        for p in positions:
            bin_idx = binner.bin_idx(p)
            try:
                occupancy[bin_idx] += bin_len
            except Exception as e:
                print(p, bin_idx, binner.num_bins, binner.maze_size)
                raise e

        # set rarely visited bins to zero
        occupancy[occupancy < (min_thr * 1000)] = 0.

        if smooth_sd is not None:
            occupancy = gaussian_filter(occupancy, smooth_sd)

        return occupancy / 1000  # ms to seconds


class FiringRateMap(Map):
    def __init__(self, spike_train=None, positions=None, maze_size=None, bin_size=None, bin_len=None, occ_thr=.256, smooth_sd=3, mmap=None, occ=None):
        """ Produce firing rate map for the given single cell spike train.

            Args:
                positions - array of positions of shape (n,1) or (n,2)
                maze_size - size of the maze in the same units as positions
                bin_size - size of spatial bins in the same units as positions
                bin_len - duration of temporal bins in ms
                occ_thr - zero-out bins with occupancy smaller than this
                smooth_sd - SD in bins for gaussian smoothing
                mmap - map to use as firing rate map, if provided previous args should be None
                occ - occupancy map to use if `mmap` is provided
            """
        if mmap is not None:
            __fr_map = mmap
            __raw_fr_map = None
            __occupancy = occ
            binner = None
        else:
            if maze_size is None:
                maze_size = np.ceil(positions.max(axis=0)).astype(int) + 1
            __fr_map, __occupancy = self.__compute_frm(spike_train, positions, maze_size, bin_size, bin_len, occ_thr, smooth_sd=smooth_sd, return_occupancy=True)
            __raw_fr_map = self.__compute_frm(spike_train, positions, maze_size, bin_size, bin_len, occ_thr, smooth_sd=0, return_occupancy=False)
            binner = __occupancy.binner
        super().__init__(__fr_map)
        self.__occupancy = __occupancy
        self.bin_size = bin_size
        self.__eps = 1e-15
        self.__I_sec = None
        self.__I_spike = None
        self.__sparsity = None
        self.__frs = None  # mean, median, max
        self.__raw_fr_map = __raw_fr_map
        self.binner = binner

    @property
    def occupancy(self):
        """ Occupancy map. """
        return self.__occupancy

    @property
    def peak_fr(self):
        """ Peak firing rate in Hz. """
        if self.__frs is None:
            self.__frs = self.__compute_frs()
        return self.__frs[2]

    @property
    def mean_fr(self):
        """ Mean firing rate in Hz. """
        if self.__frs is None:
            self.__frs = self.__compute_frs()
        return self.__frs[0]

    def __calc_I_sec(self, fr_map):
        good = np.logical_and(self.__good_idx(fr_map, self.__eps), self.__good_idx(self.__occupancy.map, self.__eps))
        frm = fr_map[good]
        occ = self.__occupancy.map_prob[good]
        mfr = np.maximum(self.mean_fr, self.__eps)
        return np.sum(frm * np.log2(frm / mfr) * occ)

    @property
    def I_sec(self):
        """ Information of the firing rate map in bits/s. """
        if self.__I_sec is None:
            self.__I_sec = self.__calc_I_sec(self.map)
        return self.__I_sec

    @property
    def I_spike(self):
        """ Information of the firing rate map in bits/spike. """
        if self.__I_spike is None:
            self.__I_spike = self.I_sec / np.maximum(self.mean_fr, self.__eps)
        return self.__I_spike

    def I_sec_shuffle(self, n_sh=100):
        idx_sh = [np.random.permutation(self.map.size) for _ in range(n_sh)]
        frm_sh = [self.map.flatten()[idx].reshape(self.map.shape)
                  for idx in idx_sh]
        return np.array([self.__calc_I_sec(frm) for frm in frm_sh])

    def I_spike_shuffle(self, n_sh=100):
        I_sec_sh = self.I_sec_shuffle(n_sh)
        mfr = np.maximum(self.mean_fr, self.__eps)
        return I_sec_sh / mfr

    @property
    def sparsity(self):
        """ Sparsity of the firing rate map. """
        if self.__sparsity is None:
            occ_prob = self.__occupancy.map_prob
            frm_good_idx = self.__good_idx(self.map)
            occ_good_idx = self.__good_idx(occ_prob)
            good_idx = np.logical_and(frm_good_idx, occ_good_idx)

            frm = self.map[good_idx]
            occ = occ_prob[good_idx]

            self.__sparsity = np.sum(frm * occ) ** 2 / np.sum(frm ** 2 * occ)
        return self.__sparsity

    @property
    def coherence(self):
        r = np.corrcoef(self.__raw_fr_map.flatten(), self.map.flatten())[0, 1]
        return .5 * np.log((1 + r) / (1 - r))

    @cached_property
    def center(self):
        if (self.map != 0).sum() == 0:
            return None
        return np.squeeze(np.where(self.map == self.peak_fr))

    def correlate(self, other: Map, normalized=False):
        self_ok = self.__good_idx(self.map) & (self.occupancy.map > 0)
        other_ok = self.__good_idx(other.map) & (other.occupancy.map > 0)
        both_ok = (self_ok) & (other_ok)
        if normalized:
            sm = self.map_prob[both_ok].flatten()
            om = other.map_prob[both_ok].flatten()
        else:
            sm = self.map[both_ok].flatten()
            om = other.map[both_ok].flatten()
        return np.corrcoef(sm, om)[0, 1]

    def z_score(self, occ_thr=0.):
        """ Z-score firing rate map.

            Args:
                occ_thr - consider only spatial bins with occupancy above this value
            Return:
                z-scored firing rate map, np.ndarray
        """
        idx = self.occupancy.map > occ_thr
        m = self.map[idx].mean()
        s = self.map[idx].std()
        return (self.map - m) / s

    def __compute_frs(self):
        """ Compute mean, median and max firing rates from given firing rate map.
            Ignore nans, infs and values smaller than eps.

            Return:
                (mean, median, max) FR in Hz
        """
        not_nan = np.logical_not(np.isnan(self.map))
        not_inf = np.logical_not(np.isinf(self.map))
        good = np.logical_and(not_nan, not_inf)
        good = np.logical_and(good, self.map >= self.__eps)
        frm = self.map[good]
        if frm.size == 0:
            return 0, 0, 0
        return frm.mean(), np.median(frm), frm.max()

    @staticmethod
    def __compute_frm(spike_train, positions, maze_size, bin_size, bin_len, occ_thr, smooth_sd=3, return_occupancy=False):
        """ Produce firing rate map for the given single cell spike train.

            Args:
                positions - array of positions of shape (n,1) or (n,2)
                maze_size - size of the maze in the same units as positions
                bin_size - size of spatial bins in the same units as positions
                bin_len - duration of temporal bins in ms
                occ_thr - zero-out bins with occupancy smaller than this
                smooth_sd - SD in bins for gaussian smoothing
                return_occupancy - whether to return occupancy map
            Return:
                firing rate map - matrix with firing rate (Hz) in each spatial bin
        """
        assert len(spike_train) == len(positions)
        occupancy = OccupancyMap(positions, maze_size, bin_size, bin_len, occ_thr, smooth_sd)
        frm = np.zeros_like(occupancy.map)

        for p, sn in zip(positions, spike_train):
            bin_idx = occupancy.binner.bin_idx(p)
            frm[bin_idx] += sn

        frm = frm / occupancy.map
        frm[np.isnan(frm)] = 0
        frm[np.isinf(frm)] = 0
        frm = gaussian_filter(frm, smooth_sd)
        if return_occupancy:
            return frm, occupancy
        else:
            return frm

    @staticmethod
    def __good_idx(m, eps=None):
        """ Get indices of elements that are not Nan, Inf and that are >= `eps`. """
        not_nan = np.logical_not(np.isnan(m))
        not_inf = np.logical_not(np.isinf(m))
        good = np.logical_and(not_nan, not_inf)
        if eps is not None:
            return np.logical_and(good, m >= eps)
        else:
            return good


class PopulationVectors:
    def __init__(self, fr_maps, z_score=False):
        """ Get population vectors from provided firing rate maps.

            Args:
                fr_maps - list of FiringRateMap, all of the same shape
            Return:
                population vectors - np.array of shape (*(fr_maps[0].shape), len(fr_maps))
        """
        self.pvs = np.stack([frm.map for frm in fr_maps], axis=-1)
        if z_score:
            m = self.pvs.mean(axis=(0, 1))
            s = self.pvs.std(axis=(0, 1))
            self.pvs = (self.pvs - m) / s

    def from_pvs(self, pvs, z_score=False):
        self.pvs = pvs
        """ Get population vectors from provided population vectors.

            Args:
                pvs - population vectors, 3D np.array
            Return:
                population vectors
        """
        if z_score:
            m = self.pvs.mean(axis=(0, 1))
            s = self.pvs.std(axis=(0, 1))
            self.pvs = (self.pvs - m) / s
        return self

    @property
    def spatial_shape(self):
        return self.pvs.shape[:-1]

    @property
    def num_cells(self):
        return self.pvs.shape[-1]

    def measure(self, other, measure_fun):
        """ Compute measure `measure_fun` between all pairs of population vectors.

            Args:
                other - other population vectors
                measure_fun - function that accepts two vectors and returns a number
            Return:
                list of measure values
        """
        assert self.pvs.shape == other.pvs.shape
        spatial_shape = self.pvs.shape[:-1]
        measure = np.array([measure_fun(self.pvs[idx], other.pvs[idx]) for idx in np.ndindex(spatial_shape)])
        return measure[~np.isnan(measure)]

    def correlation(self, other):
        """ Compute Pearson correlation coefficients between all pairs of population vectors.

            Args:
                other - other population vectors
            Return:
                correlation coefficients - list of correlation coefficients
        """
        return self.measure(other, lambda pv1, pv2: np.corrcoef(pv1, pv2)[0,1])

    def cosine_distance(self, other):
        """ Compute correlation coefficients between all pairs of population vectors.

            Args:
                other - other population vectors
            Return:
                list of cosine distances
        """
        return self.measure(other, cosine)

