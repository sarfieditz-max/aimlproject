"""Prediction script for the K-Means shopper-segment model."""
import json, os
import numpy as np
import joblib
import pandas as pd
from kmeans_prep import add_features          # kmeans_prep.py sits in the same folder

HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = joblib.load(os.path.join(HERE, 'kmeans_segmentation_pipeline.joblib'))
with open(os.path.join(HERE, 'segment_info.json')) as f:
    _INFO = json.load(f)

# optional fields get a neutral default so the form can stay short
DEFAULTS = {'PageValues': 0.0, 'Month': 'May', 'VisitorType': 'Returning_Visitor', 'Weekend': False,
            'OperatingSystems': 2, 'Browser': 2, 'Region': 1, 'TrafficType': 2}
REQUIRED = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
            'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'SpecialDay']


def predict_segment(session: dict) -> dict:
    """session = one visit, using the same field names as the original dataset (without Revenue).
    Returns the segment plus how close the session is to EACH segment centre (so borderline sessions are visible)."""
    missing = [c for c in REQUIRED if c not in session]
    if missing:
        raise ValueError(f'Missing fields: {missing}')
    X = add_features(pd.DataFrame([{**DEFAULTS, **session}]))
    Z = _PIPE[:-1].transform(X)                              # same cap/scale steps as in training
    dist = _PIPE[-1].transform(Z)[0]                         # distance to every cluster centre
    cluster = int(np.argmin(dist))                           # nearest centre = the K-Means rule
    inv = 1.0 / (dist + 1e-9)
    closeness = {_INFO['segments'][str(c)]['name']: float(inv[c] / inv.sum()) for c in range(len(dist))}   # simple "how close" share
    seg = _INFO['segments'][str(cluster)]
    return {'cluster': cluster, 'segment': seg['name'], 'segment_training_purchase_rate': seg['training_purchase_rate'],
            'overall_purchase_rate': _INFO['training_overall_purchase_rate'], 'advice': seg['description'],
            'closeness': closeness, 'distances': {_INFO['segments'][str(c)]['name']: round(float(d), 3) for c, d in enumerate(dist)}}


def segment_table() -> pd.DataFrame:
    """Typical session of each segment (training averages, original units) - handy for explaining the output."""
    rows = []
    for s in _INFO['segments'].values():
        rows.append({'segment': s['name'], 'share_of_sessions': s['training_share'],
                     'purchase_rate': s['training_purchase_rate'], **s['typical_session']})
    return pd.DataFrame(rows).sort_values('ProductRelated').reset_index(drop=True)


def segment_shares(raw_sessions: pd.DataFrame) -> pd.DataFrame:
    """Monitoring helper: share of new sessions per segment vs the share at training time."""
    labels = _PIPE.predict(add_features(raw_sessions))
    now = pd.Series(labels).value_counts(normalize=True)
    rows = [{'segment': s['name'], 'training_share': s['training_share'], 'new_share': round(float(now.get(int(c), 0.0)), 4)}
            for c, s in _INFO['segments'].items()]
    out = pd.DataFrame(rows); out['difference'] = (out.new_share - out.training_share).round(4)
    return out


if __name__ == '__main__':
    print(predict_segment({'Administrative': 3, 'Administrative_Duration': 80, 'Informational': 0, 'Informational_Duration': 0,
                           'ProductRelated': 45, 'ProductRelated_Duration': 1800, 'BounceRates': 0.005, 'ExitRates': 0.02,
                           'SpecialDay': 0.0, 'PageValues': 12.0}))
