"""EEGMMIDB re-epoched for the speed-dial task: imagery commands + explicit REST (idle) class,
with an optional Muse-like 4-channel proxy (AF7, AF8, TP7, TP8 ~ Muse AF7/AF8/TP9/TP10)."""
import numpy as np, mne
from mne.datasets import eegbci
mne.set_log_level("ERROR")
MUSE_PROXY = ["AF7", "AF8", "TP7", "TP8"]


def load_eegbci_with_rest(subjects, tmin=0.0, tmax=4.0, fmin=1.0, fmax=40.0, resample=128, path="data/eegbci"):
    Xs, ys, gs = [], [], []
    for s in subjects:
        for runs, mapping in (([4, 8, 12], {"T0": "rest", "T1": "left_hand", "T2": "right_hand"}),
                              ([6, 10, 14], {"T0": "rest", "T1": "hands", "T2": "feet"})):
            fn = eegbci.load_data(s, runs, path=path, update_path=False, verbose=False)
            raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fn])
            eegbci.standardize(raw)
            raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")
            raw.filter(fmin, fmax, fir_design="firwin", verbose=False)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            ev_id = {mapping[k]: v for k, v in event_id.items() if k in mapping}
            ep = mne.Epochs(raw, events, ev_id, tmin, tmax, baseline=None, preload=True, picks="eeg", verbose=False)
            ep.resample(resample, verbose=False)
            inv = {v: k for k, v in ev_id.items()}
            Xs.append(ep.get_data(copy=True)); ys.append(np.array([inv[e] for e in ep.events[:, 2]])); gs.append(np.full(len(ep), s))
            ch = ep.ch_names; sf = ep.info["sfreq"]
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), ch, sf
