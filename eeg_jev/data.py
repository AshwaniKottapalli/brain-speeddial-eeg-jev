"""Load and epoch motor-imagery EEG from PhysioNet EEGMMIDB and BCI IV 2a.

Returns numpy arrays: X (n_trials, n_channels, n_times), y (str labels), groups (subject id).
"""
from __future__ import annotations
import numpy as np
import mne
from mne.datasets import eegbci

mne.set_log_level("ERROR")

# EEGMMIDB: runs 4,8,12 -> T1=left fist, T2=right fist ; runs 6,10,14 -> T1=both fists, T2=both feet
EEGBCI_RUNS_LR = [4, 8, 12]
EEGBCI_RUNS_HF = [6, 10, 14]
MOTOR_CHANNELS = ["FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                  "CP3", "CP1", "CPz", "CP2", "CP4"]


def load_eegbci(subjects, tmin=-1.0, tmax=4.0, fmin=1.0, fmax=40.0, resample=128, path="data/eegbci",
                classes=("left_hand", "right_hand", "hands", "feet")):
    Xs, ys, gs = [], [], []
    for s in subjects:
        for runs, mapping in ((EEGBCI_RUNS_LR, {"T1": "left_hand", "T2": "right_hand"}),
                              (EEGBCI_RUNS_HF, {"T1": "hands", "T2": "feet"})):
            if not any(v in classes for v in mapping.values()):
                continue
            fnames = eegbci.load_data(s, runs, path=path, update_path=False, verbose=False)
            raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames]
            raw = mne.concatenate_raws(raws)
            eegbci.standardize(raw)
            raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")
            raw.filter(fmin, fmax, fir_design="firwin", verbose=False)
            raw.set_eeg_reference("average", projection=False, verbose=False)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            ev_id = {mapping[k]: v for k, v in event_id.items() if k in mapping and mapping[k] in classes}
            epochs = mne.Epochs(raw, events, ev_id, tmin, tmax, baseline=None, preload=True,
                                picks="eeg", verbose=False)
            epochs.resample(resample, verbose=False)
            X = epochs.get_data(copy=True)
            inv = {v: k for k, v in ev_id.items()}
            y = np.array([inv[e] for e in epochs.events[:, 2]])
            Xs.append(X); ys.append(y); gs.append(np.full(len(y), s))
            ch_names = epochs.ch_names; sfreq = epochs.info["sfreq"]
    X = np.concatenate(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
    return X, y, g, ch_names, sfreq, tmin


def load_bnci(subjects, tmin=-1.0, tmax=4.0, fmin=1.0, fmax=40.0, resample=128):
    from moabb.datasets import BNCI2014_001
    ds = BNCI2014_001()
    Xs, ys, gs = [], [], []
    for s in subjects:
        data = ds.get_data(subjects=[s])[s]
        for sess_name, sess in data.items():
            for run_name, raw in sess.items():
                raw = raw.copy().load_data()
                raw.pick("eeg")
                raw.filter(fmin, fmax, fir_design="firwin", verbose=False)
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                ev_id = {k: v for k, v in event_id.items() if k in ("left_hand", "right_hand", "feet", "tongue")}
                epochs = mne.Epochs(raw, events, ev_id, tmin, tmax, baseline=None, preload=True, verbose=False)
                epochs.resample(resample, verbose=False)
                X = epochs.get_data(copy=True)
                inv = {v: k for k, v in ev_id.items()}
                y = np.array([inv[e] for e in epochs.events[:, 2]])
                Xs.append(X); ys.append(y); gs.append(np.full(len(y), s))
                ch_names = epochs.ch_names; sfreq = epochs.info["sfreq"]
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), ch_names, sfreq, tmin


def cache_dataset(name, loader, **kw):
    import os
    fn = f"cache/{name}.npz"
    if os.path.exists(fn):
        d = np.load(fn, allow_pickle=True)
        return d["X"], d["y"], d["g"], list(d["ch"]), float(d["sfreq"]), float(d["tmin"])
    X, y, g, ch, sfreq, tmin = loader(**kw)
    np.savez_compressed(fn, X=X.astype(np.float32), y=y, g=g, ch=np.array(ch), sfreq=sfreq, tmin=tmin)
    return X, y, g, ch, sfreq, tmin
