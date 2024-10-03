from data_util import load, DAYS, z_score_bins, sess_nums_from_desen
from tqdm import tqdm
from pandas import DataFrame as DF, concat, read_csv
import numpy as np
from scipy.ndimage import gaussian_filter1d


def get_sess(dn):
    path = f"data/raw/charlotte/{dn}/{dn}.desen"
    desen = read_csv(path, delimiter=' ',
                     names=["sess_type", "ll", "num"],
                     dtype={"sess_type": str, "ll": str})
    return {i: np.array(i) for i in range(1, len(desen)+1)}


def bin_idx_to_ts(bin_idx, bl, sp):
    df = np.diff(np.concatenate([[0], bin_idx, [0]]))
    bi = np.where(df != 0)[0].reshape(-1, 2)
    return (bi * bl / sp).astype(np.int64)


# Results:
#   * for CA1 in data/raw/charlotte/{day}/{day}_{sess}.hse.p1
#   * for mEC in data/raw/charlotte/{day}/{day}_{sess}.hse.pe

if __name__ == "__main__":
    # as in O'Neil 2016 Science paper
    # (https://www-science-org.libraryproxy.ista.ac.at/doi/suppl/10.1126/science.aag2787/suppl_file/oneill.sm.pdf)
    SP = .05  # sampling period in ms
    CTs = ["p1", "pe"]
    THR = 3  # z-score threshold
    BL = 1.5  # ms
    SIGMA = 15
    SIGMA = 15 / BL
    MIN_DUR = int(75 // BL)
    MIN_DUR = int(50 // BL)
    MAX_DUR = int(750 // BL)
    MAX_DUR = int(1000 // BL)
    MIN_DIST = int(50 // BL)
    MIN_SPIKES = 5
    MIN_CELLS = 4
    for DN in tqdm(DAYS):
        #if DN in ["mjc73-3009-0525", "mjc111-2106-1743", "mjc134-1303-0735", "mjc111-2406-1337", "mjc111-2306-1839", "mjc134-0903-0734", "mjc73-2709-0425"]:
        #if DN != "mjc73-2709-0425":
        #    continue
        print(DN)
        sess_nums = sess_nums_from_desen(DN)
        print(sess_nums)
        for sess, sn in sess_nums.items():
            #if sess != "LEARNING":
            #    continue
            try:
                data = load(DN, sess, sess_nums, BL, SP)
                for CT in CTs:
                    bins_ct = data.bins[data.ct == CT].astype(np.float32)
                    pop = gaussian_filter1d(bins_ct.sum(axis=0), SIGMA)
                    pop_z = z_score_bins(pop.reshape(1, -1)).flatten()
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
                    np.savetxt(f"data/raw/charlotte/{DN}/{DN}_{sn}.hse.{CT}",
                               hse_ts, fmt="%d")
                    print(f"{DN} {sess}", CT, hse_ts.shape[0], "HSEs")
                    del hse_begins, hse_peaks, hse_ends, all_durs
                del data
            except Exception as e:
                print(e)
                print(DN, sess, sn)
                continue
