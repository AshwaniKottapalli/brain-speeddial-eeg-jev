"""Build cached epochs + JSON features for both datasets. Run once."""
import sys, json, numpy as np
sys.path.insert(0, ".")
from eeg_jev.data import load_eegbci, load_bnci, cache_dataset
from eeg_jev.features import epoch_features, round_feats, feature_vector

which = sys.argv[1] if len(sys.argv) > 1 else "eegbci"
if which == "eegbci":
    X, y, g, ch, sfreq, tmin = cache_dataset("eegbci_s1-10", load_eegbci, subjects=list(range(1, 11)))
else:
    X, y, g, ch, sfreq, tmin = cache_dataset("bnci_s1-3", load_bnci, subjects=[1, 2, 3])
print(which, X.shape, dict(zip(*np.unique(y, return_counts=True))), "subjects", np.unique(g), "sfreq", sfreq)
feats = [round_feats(epoch_features(x, ch, sfreq, tmin)) for x in X]
F = np.stack([feature_vector(f) for f in feats])
json.dump({"feats": feats, "y": y.tolist(), "g": g.tolist()}, open(f"cache/{which}_feats.json", "w"))
np.save(f"cache/{which}_F.npy", F)
print("feature vector dim", F.shape, "example:\n", json.dumps(feats[0], indent=1)[:1200])
