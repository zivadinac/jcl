from data_util import load, sess_nums_from_desen, DAYS, cut_probe
from tqdm import tqdm
from pandas import DataFrame as DF, concat
import numpy as np
from jcl.analysis import FiringRateMap
import plotly.express as px
from scipy.stats import kstest
from scipy.spatial.distance import euclidean
from skimage.filters import threshold_otsu
from datetime import datetime


def load_sess(sess, dn, sess_nums, bl, sp):
    ddata = load(dn, sess, sess_nums, bl, sp)
    idx = cut_probe(ddata.whl, ddata.sb, bl)
    bins, whl = ddata.bins[:, idx], ddata.whl[idx]
    return bins, ddata.ct, whl, ddata.goals


# Classify cells as spatially selective if:
#      sparsity is <= P1_THR for p1
#      sparsity is <= PE_THR for pe
# TODO proper control with a GLM / subsampling (as in the original paper)

if __name__ == "__main__":
    BL = 102.4  # 25.6  # ms
    BS = 2  # cm
    SP = .05  # sampling period in ms
    MS = None  # (155, 155)
    # sparisity
    P1_THR = .3
    PE_THR = .9
    # I_spike
    P1_THR = 1.5
    PE_THR = 1.1
    spat = []
    for DN in tqdm(DAYS):
        sess_nums = sess_nums_from_desen(DN)
        sess = "OPEN-FIELD" if sess_nums["OPEN-FIELD"] else "PRE-PROBE"
        #sess = "PRE-PROBE"
        bins_pre, ct, whl_pre, goals = load_sess(sess, DN, sess_nums, BL, SP)
        frms_pre = [FiringRateMap(c, whl_pre, MS, BS, BL)
                    for c in bins_pre]
        spat_d = DF({"sparsity": [fm.sparsity for fm in frms_pre],
                     "coherence": [fm.coherence for fm in frms_pre],
                     "I_spike": [fm.I_spike for fm in frms_pre],
                     "mean_FR": [fm.mean_fr for fm in frms_pre],
                     "des": ct, "day": DN})
        ss = []
        for (sp, des) in zip(spat_d["I_spike"], spat_d["des"]):
            if 'b' in des:
                ss.append(False)
            elif des == "pe":
                ss.append(sp >= PE_THR)
            elif des == "p1":
                ss.append(sp >= P1_THR)
            else:
                raise Exception(f"Unknown cell type {des}")
        """
        for (sp, des) in zip(spat_d["sparsity"], spat_d["des"]):
            if 'b' in des:
                ss.append(False)
            elif des == "pe":
                ss.append(sp <= PE_THR)
            elif des == "p1":
                ss.append(sp <= P1_THR)
            else:
                raise Exception(f"Unknown cell type {des}")
        """
        np.savetxt(f"data/raw/charlotte/{DN}/{DN}.ss", ss, fmt="%d")
        spat.append(spat_d)
        print(DN, np.mean(ss))
    spat = concat(spat)
    dt = datetime.now()
    px.histogram(spat, x="sparsity", color="des", title=f"sparsity {dt}").write_html("reports/spatial_properties/sparsity.html")
    px.histogram(spat, x="coherence", color="des", title=f"coherence {dt}").write_html("reports/spatial_properties/coherence.html")
    px.histogram(spat, x="I_spike", color="des", title=f"I/spike {dt}").write_html("reports/spatial_properties/I_spike.html")
