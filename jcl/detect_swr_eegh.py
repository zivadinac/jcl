from argparse import ArgumentParser
import numpy as np
from elephant.signal_processing import butter
import plotly.express as px


def z_score(x):
    return (x-x.mean()) / x.std()

def bp_filter(eegh, hp, lp, fs=5000):
    return butter(eegh, hp, lp, sampling_frequency=fs)

def moving_average(a, n=32):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

def find_swrs(eegh, tets, ref_tet,
              peak_std_thr=7, end_std_thr=1.2,
              hp=150, lp=250, peak_dist_s=.05,
              eegh_sr=5000, dat_sr=20000,
              basename=None):
    assert eegh.ndim == 2
    assert len(tets) > 0
    assert dat_sr > eegh_sr
    assert lp > hp
    assert peak_std_thr > end_std_thr

    rt_eegh = bp_filter(eegh[:, ref_tet], hp, lp, eegh_sr)
    f_eegh = [bp_filter(eegh[:, t], hp, lp, eegh_sr) - rt_eegh for t in tets]
    p_eegh = [moving_average(np.abs(f)**2) for f in f_eegh]
    max_p_tet = np.argmax(np.mean(p_eegh, axis=1))
    #p_eegh = np.max([moving_average(np.abs(f)**2) for f in f_eegh], axis=0)
    p_eegh = p_eegh[max_p_tet]
    zf_eegh = z_score(p_eegh)

    # find SWR peaks
    swr_peaks = []
    prev_i = -1
    for i in np.where(zf_eegh >= peak_std_thr)[0]:
        if i - prev_i > (peak_dist_s * eegh_sr) or len(swr_peaks) == 0:
            swr_peaks.append([])
        swr_peaks[-1].append(i)
        prev_i = i
    swr_peaks = [p[np.argmax(p_eegh[p])] for p in swr_peaks]
    if len(swr_peaks) == 0:
        return np.empty((0, 3))

    # find SWR limits
    swr_begins, swr_ends = [], []
    for p in swr_peaks:
        b = p - np.argmax(np.flip(zf_eegh[np.maximum(p-eegh_sr, 0):p]) < end_std_thr) + 1
        e = p + np.argmax(zf_eegh[p:p+eegh_sr] < end_std_thr) - 1
        if len(swr_ends) > 0:
            b = np.maximum(b, swr_ends[-1] + 1)
        swr_begins.append(b)
        swr_ends.append(e)
    return np.stack([swr_begins, swr_peaks, swr_ends], axis=1) * (dat_sr / eegh_sr)


args = ArgumentParser()
args.add_argument("basename")
args.add_argument("--hp", default=150, type=int, help="High pass SWR frequency (default 150Hz).")
args.add_argument("--lp", default=250, type=int, help="Low pass SWR frequency (default 250Hz).")
args.add_argument("--peak_std_thr", default=7, type=float, help="Threshold for SWR peak detection (default 7).")
args.add_argument("--end_std_thr", default=1.2, type=float, help="Threshold for SWR end detection (default 1.2).")
args.add_argument("--peak_dist_s", default=0.05, type=float, help="Minimum distance between two SWR peaks (in seconds, default 0.05 == 50ms).")
args.add_argument("--eegh_sr", default=5000, type=int, help="Sampling rate of .eegh file (default 5kHz).")
args.add_argument("--dat_sr", default=20000, type=int, help="Sampling rate of orignal data (default 20kHz).")
args = args.parse_args()
#args.basename = "~/workspace/ca1_mec_goals_reactivation/data/raw/charlotte/mjc134-1203-0834_8/mjc134-1203-0834_8"

print("=========================================================")
print(f"Processing {args.basename}.")
try:
    desel = np.loadtxt(f"{args.basename}.desel", dtype=str)
except:
    bn = args.basename.split('_')[0]
    print(f"Cannot find {args.basename}. Trying with {bn}.")
    desel = np.loadtxt(f"{bn}.desel", dtype=str)

eegh = np.fromfile(f"{args.basename}.eegh", dtype=np.int16).reshape(-1, 16)
tets = np.where(desel == '1')[0]
ref_tet = np.where(desel != '1')[0][-1]
swrs = find_swrs(eegh, tets, ref_tet=ref_tet, **vars(args))
np.savetxt(f"{args.basename}.sw", swrs, fmt="%d", delimiter=' ')
print(f"Finished {args.basename}.")
print("=========================================================")
