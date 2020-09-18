import numpy as np
from scipy.interpolate import interp1d, griddata

def align_measured_data(population_vectors, positions, num_bins, speed=None):
    l = positions.shape[0] // num_bins
    if l < population_vectors.shape[1]:
        pv = population_vectors[:, 0:l]
        pos = positions[0:l*num_bins, :]
        s = speed[0:l*num_bins] if speed is not None else None
    else:
        ll = population_vectors.shape[1] * num_bins
        pv = population_vectors
        pos = positions[0:ll, :]
        s = speed[0:ll] if speed is not None else None

    pos = pos.reshape(-1, num_bins, 2).mean(axis=1)
    s = s.reshape(-1, num_bins).mean(axis=1) if speed is not None else None

    if s is not None:
        return pv, pos, s
    else:
        return pv, pos

#pv, pp, ss = align_measured_data(fr, p, 4, speed=s)

def __interpolate_linear_position(position, unknown_val=-1):
    ok_inds = np.where(position != unknown_val)[0]
    f = interp1d(ok_inds, position[ok_inds])
    return f(np.arange(len(position))

def __interpolate_2d_position(x, y, unknown_val=-1):
    def _no_unknown_val(x):
        return np.all(x != unknown_val)

    positions = np.vstack(x, y)
    ok_inds = np.where([_no_unknown_val(p) for p in positions])
    all_inds = np.arange(len(positions))
    return griddata((ok_inds, ok_inds), positions[ok_inds], (all_inds, all_inds))
    
def interpolate_position(x, y=None, unknown_val=-1):
    if y is None:
        return __interpolate_linear_position(x, unknown_val)
    
    return __interpolate_2d_position(x, y, unknown_val)

