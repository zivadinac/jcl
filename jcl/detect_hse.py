from argparse import ArgumentParser
import numpy as np
from scipy.ndimage import gaussian_filter1d
from load import bins as load_bins, cell_types


if __name__ == "__main__":
    # as in O'Neil 2016 Science paper
    # (https://www-science-org.libraryproxy.ista.ac.at/doi/suppl/10.1126/science.aag2787/suppl_file/oneill.sm.pdf)

    args = ArgumentParser()
    args.add_argument("sess_basename")
    args.add_argument("--sampling_rate_hz", default=20_000, type=int)
    args.add_argument("--cell_type", default="p1")
    args.add_argument("--sd_thr", default=3, type=float)
    args.add_argument("--bin_len", default=1.5, type=float)
    args.add_argument("--sigma_ms", default=15, type=float)
    args.add_argument("--min_dur_ms", default=50, type=float)
    args.add_argument("--max_dur_ms", default=750, type=float)
    args.add_argument("--min_dist_ms", default=50, type=float)
    args.add_argument("--min_cells", default=5, type=int)
    args.add_argument("--min_spikes", default=5, type=int)
    args = args.parse_args()

    SP = 1000. / args.sampling_rate_hz  # sampling period in ms
    CT = args.cell_type
    THR = args.sd_thr  # z-score threshold
    BL = args.bin_len
    SIGMA = args.sigma_ms / BL
    MIN_DUR = int(args.min_dur_ms // BL)
    MAX_DUR = int(args.max_dur_ms // BL)
    MIN_DIST = int(args.min_dist_ms // BL)
    MIN_CELLS = args.min_cells
    MIN_SPIKES = args.min_spikes
    basename = args.sess_basename

    try:
        ct = cell_types(basename + ".des")[1:]
    except FileNotFoundError:
        pos = basename.rfind('_')
        ct = cell_types(basename[:pos] + ".des")[1:]

    bins = load_bins(basename + ".res", basename + ".clu", BL, SP)
    bins_ct = bins[ct == CT].astype(np.float32).toarray()
    pop = gaussian_filter1d(bins_ct.sum(axis=0), SIGMA)
    pop_z = (pop - pop.mean()) / pop.std()
    hse_peaks = []
    prev_i = -1
    for i in np.where(pop_z >= THR)[0]:
        if len(hse_peaks) == 0 or i - prev_i > MIN_DIST:
            hse_peaks.append([])
        hse_peaks[-1].append(i)
        prev_i = i
    hse_peaks = [p[np.argmax(pop_z[p])] for p in hse_peaks]
    hse_begins, hse_ends, all_durs = [], [], []
    for p in hse_peaks:
        if p <= 1:
            continue
        pre_peak = pop_z[np.maximum(0, p-MAX_DUR):p]
        b = p - np.argmin(np.flip(pre_peak) > 1)
        post_peak = pop_z[p:p+MAX_DUR]
        e = p + np.argmin(post_peak > 1)
        assert b <= p <= e
        b, e = int(b), int(e)
        if len(hse_ends) > 0 and hse_ends[-1] >= b:
            b = hse_ends[-1] + 1
            if e - b <= 1:
                continue
        dur = e - b
        assert dur >= 0
        spk = bins_ct[:, b:e].sum()
        cells = (bins_ct[:, b:e].sum(axis=1) > 0).sum()
        all_durs.append((dur, spk, cells))
        if dur < MIN_DUR:
            continue
        if dur > MAX_DUR:
            continue
        if spk < MIN_SPIKES:
            continue
        min_cells = np.minimum(MIN_CELLS, int(.1 * bins_ct.shape[0]))
        if cells < min_cells:
            continue
        hse_begins.append(b)
        hse_ends.append(e)
    hse_ts = (np.stack([hse_begins, hse_ends], axis=1) * BL / SP).astype(int)
    np.savetxt(basename + f".hse.{CT}",
               hse_ts, fmt="%d")
    print(f"{basename}", CT, hse_ts.shape[0], "HSEs")
