"""
BatteryBench — Data Loader
XJTU Battery Dataset · SOH Estimation

Dataset:
    6 Batches x 55 batteries x 131-1301 cycles
    Each .mat: summary (cycle stats) + data (raw signals)

SOH = discharge_capacity_Ah[i] / discharge_capacity_Ah[0] x 100

Split (cross-batch generalization):
    Train : Batch-1 (2C) + Batch-2 (3C) + Batch-3 (R2.5) + Batch-4 (R3)
    Val   : Batch-5 (RW)
    Test  : Batch-6 (Satellite)

Input: 9 summary features per cycle, sliding window of 32 cycles
Run  : python src/battery_data_loader.py
"""

import os
import sys
import pickle
import numpy as np
import scipy.io
from sklearn.preprocessing import MinMaxScaler

# Paths
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_PATH = os.path.join(CACHE_DIR, 'batterybench.pkl')

# Config
SEED     = 42
SEQ_LEN  = 32
STRIDE   = 1
EOL_SOH  = 80.0

BATCH_CONFIG = {
    'Batch-1': {'protocol': '2C',        'split': 'trainval'},
    'Batch-2': {'protocol': '3C',        'split': 'trainval'},
    'Batch-3': {'protocol': 'R2.5',      'split': 'trainval'},
    'Batch-4': {'protocol': 'R3',        'split': 'trainval'},
    'Batch-5': {'protocol': 'RW',        'split': 'test'},
    
}

SKIP_FILES = ['Temperature_Compensation_Data.mat']

FEATURE_NAMES = [
    'discharge_capacity_Ah',
    'charge_capacity_Ah',
    'discharge_power_Wh',
    'charge_power_Wh',
    'charge_median_voltage',
    'discharge_median_voltage',
    'charge_mean_voltage',
    'discharge_mean_voltage',
    'cycle_norm',
]
N_FEATURES = len(FEATURE_NAMES)


def load_battery(path):
    mat     = scipy.io.loadmat(path, simplify_cells=True)
    keys    = [k for k in mat.keys() if not k.startswith('__')]
    sum_key = next((k for k in keys if 'sum' in k.lower()), None)
    if not sum_key:
        return None, None, None

    s = mat[sum_key]
    n = int(s['cycle_life'])

    cap = np.array(s['discharge_capacity_Ah'], dtype=np.float32)
    # Use max of first 5 cycles as reference capacity
    # Handles formation phase (capacity rise before degradation)
    # and satellite profiles where cap[0] is not representative
    # Use global max capacity as reference
    # Satellite has partial cycles — cap[0] and first few
    # cycles may not represent true full capacity
    ref_cap = float(np.max(cap))
    soh     = np.clip(cap / ref_cap * 100.0, 0.0, 100.0).astype(np.float32)

    # EOL using 5-cycle rolling median to avoid false detection
    # from partial satellite cycles
    from numpy.lib.stride_tricks import sliding_window_view
    win     = min(5, len(soh))
    pad     = np.pad(soh, (win//2, win//2), mode='edge')
    rolling = np.array([np.median(pad[i:i+win])
                        for i in range(len(soh))])
    eol     = next((i+1 for i, v in enumerate(rolling)
                    if v < EOL_SOH), n)

    cycle_norm = np.arange(1, n+1, dtype=np.float32) / n
    features = np.column_stack([
        np.array(s['discharge_capacity_Ah'],    dtype=np.float32),
        np.array(s['charge_capacity_Ah'],       dtype=np.float32),
        np.array(s['discharge_power_Wh'],       dtype=np.float32),
        np.array(s['charge_power_Wh'],          dtype=np.float32),
        np.array(s['charge_median_voltage'],    dtype=np.float32),
        np.array(s['discharge_median_voltage'], dtype=np.float32),
        np.array(s['charge_mean_voltage'],      dtype=np.float32),
        np.array(s['discharge_mean_voltage'],   dtype=np.float32),
        cycle_norm,
    ])

    fname = os.path.basename(path)
    meta = {
        'filename'   : fname,
        'name'       : fname.replace('.mat', ''),
        'cycle_life' : n,
        'eol_cycle'  : eol,
        'initial_cap': float(cap[0]),
        'final_soh'  : float(soh[-1]),
    }
    return features, soh, meta


def make_windows(features, soh, seq_len=SEQ_LEN, stride=STRIDE):
    n = len(soh)
    X, y = [], []
    start = 0
    while start + seq_len <= n:
        X.append(features[start:start + seq_len])
        y.append(soh[start + seq_len - 1])
        start += stride
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_dataset(data_dir=DATA_DIR):
    splits      = {'train': [], 'val': [], 'test': []}
    all_meta    = []
    train_feats = []
    battery_buf = {}

    print(f'\n  {"Batch":<10} {"Battery":<30} {"Cycles":>7}'
          f' {"EOL":>6} {"FinalSOH":>10}  Split')
    print('  ' + '-'*72)

    for batch_name, cfg in BATCH_CONFIG.items():
        batch_path = os.path.join(data_dir, batch_name)
        if not os.path.exists(batch_path):
            print(f'  WARNING: {batch_path} not found')
            continue

        files = sorted([f for f in os.listdir(batch_path)
                        if f.endswith('.mat') and f not in SKIP_FILES])

        for fname in files:
            features, soh, meta = load_battery(
                os.path.join(batch_path, fname))
            if features is None:
                continue

            meta.update({'batch'   : batch_name,
                         'protocol': cfg['protocol'],
                         'split'   : cfg['split']})
            all_meta.append(meta)
            battery_buf[meta['name']] = {
                'features': features,
                'soh'     : soh,
                'split'   : cfg['split'],
            }
            if cfg['split'] == 'trainval':
                train_feats.append(features)

            print(f'  {batch_name:<10} {meta["name"]:<30}'
                  f' {meta["cycle_life"]:>7} {meta["eol_cycle"]:>6}'
                  f' {meta["final_soh"]:>9.1f}%  {cfg["split"]}')

    print('\n  Fitting MinMaxScaler on training set...')
    scaler = MinMaxScaler()
    scaler.fit(np.vstack(train_feats))

    print('  Building sliding windows...')
    for name, bd in battery_buf.items():
        scaled = scaler.transform(bd['features'])
        X, y   = make_windows(scaled, bd['soh'])
        bd_split  = bd['split']
        val_ratio = 0.20   # last 20% of cycles -> val
        if bd_split == 'trainval':
            n         = len(bd['soh'])
            split_idx = int(n * (1 - val_ratio))
            tr_f = scaled[:split_idx]
            tr_s = bd['soh'][:split_idx]
            va_f = scaled[split_idx:]
            va_s = bd['soh'][split_idx:]
            if len(tr_s) >= SEQ_LEN:
                X_tr, y_tr = make_windows(tr_f, tr_s)
                splits['train'].append((X_tr, y_tr))
            if len(va_s) >= SEQ_LEN:
                X_va, y_va = make_windows(va_f, va_s)
                splits['val'].append((X_va, y_va))
        else:
            X_te, y_te = make_windows(scaled, bd['soh'])
            splits['test'].append((X_te, y_te))

    result = {}
    for split, batches in splits.items():
        if batches:
            Xa = np.vstack([b[0] for b in batches])
            ya = np.concatenate([b[1] for b in batches])
            if split == 'train':
                idx = np.random.RandomState(SEED).permutation(len(ya))
                Xa, ya = Xa[idx], ya[idx]
            result[split] = (Xa, ya)
        else:
            result[split] = (np.array([]), np.array([]))

    return result, scaler, all_meta


def load_dataset(force_rebuild=False, data_dir=DATA_DIR):
    if os.path.exists(CACHE_PATH) and not force_rebuild:
        print(f'  Loading cache: {CACHE_PATH}')
        with open(CACHE_PATH, 'rb') as f:
            splits, scaler, meta = pickle.load(f)
        for split, (X, y) in splits.items():
            if len(X):
                print(f'  {split:<6}: {X.shape}')
        return splits, scaler, meta

    print('  Building from .mat files...')
    splits, scaler, meta = build_dataset(data_dir)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump((splits, scaler, meta), f)
    print(f'\n  Cached: {CACHE_PATH}')
    return splits, scaler, meta


def to_cnn_input(X):
    """(N, SEQ_LEN, N_FEATURES) -> (N, SEQ_LEN, N_FEATURES, 1)"""
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)


if __name__ == '__main__':
    print('\nBatteryBench — Data Loader')
    print('=' * 55)
    print(f'  Dataset  : XJTU Battery (55 batteries, 6 batches)')
    print(f'  Task     : SOH regression (0-100%)')
    print(f'  Window   : {SEQ_LEN} cycles, stride={STRIDE}')
    print(f'  Features : {N_FEATURES} per cycle')
    print(f'  EOL      : SOH < {EOL_SOH}%')
    print(f'\n  Splits:')
    for b, c in BATCH_CONFIG.items():
        print(f'    {b:<10}: {c["protocol"]:<12} -> {c["split"]}')

    if not os.path.exists(os.path.join(DATA_DIR, 'Batch-1')):
        print(f'\n  ERROR: Dataset not found at {DATA_DIR}')
        print(f'  Create symlinks: ln -s /path/to/XJTU/dataset/Batch-N data/Batch-N')
        sys.exit(1)

    print('\nLoading...')
    splits, scaler, meta = load_dataset()

    print('\n-- Summary --')
    total = 0
    for split, (X, y) in splits.items():
        if len(X):
            total += len(X)
            print(f'  {split:<6}: {X.shape}  '
                  f'SOH [{y.min():.1f}-{y.max():.1f}%]  '
                  f'mean={y.mean():.1f}%')
    print(f'  Total  : {total:,} windows')

    X_tr, _ = splits['train']
    print(f'\n  LSTM/Transformer : {X_tr.shape}')
    print(f'  CNN              : {to_cnn_input(X_tr).shape}')
    print('\nData loader ready!\n')
