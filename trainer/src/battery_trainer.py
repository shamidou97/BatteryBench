import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

"""
BatteryBench — Unified Trainer
CNN vs LSTM vs Transformer · XJTU Battery SOH Estimation

Task    : Regression — predict SOH (0-100%) from cycle features
Input   : 9 summary features per cycle · sliding window of 32 cycles
Models  : CNN · LSTM · Transformer
Metrics : RMSE · MAE · MAPE · R²

Split:
    Train : Batch-1 to 4 · first 80% cycles per battery
    Val   : Batch-1 to 4 · last 20% cycles per battery
    Test  : Batch-5 (RW) · unseen protocol

Run: python src/battery_trainer.py
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# ── Path setup ────────────────────────────────────────────────
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SRC_DIR)
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, SRC_DIR)
from battery_data_loader import (load_dataset, to_cnn_input,
                                  FEATURE_NAMES, N_FEATURES, SEQ_LEN)

# ── Database config ───────────────────────────────────────────
DB_HOST     = os.environ.get('DB_HOST',     'localhost')
DB_PORT     = int(os.environ.get('DB_PORT', '3307'))
DB_USER     = os.environ.get('DB_USER',     'battuser')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'battery123')
DB_NAME     = os.environ.get('DB_NAME',     'batterybench')

# ── Config ────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS     = 80
SEED       = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# CNN
IMG_H = SEQ_LEN   # 32 cycles
IMG_W = N_FEATURES # 9 features

# Transformer
D_MODEL  = 32
N_HEADS  = 4
D_FF     = 64
N_BLOCKS = 2

# ── GPU setup ─────────────────────────────────────────────────
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f'  GPU : {gpus[0].name}')
    else:
        print('  CPU only')

# ══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════

# ── CNN ───────────────────────────────────────────────────────
def build_cnn():
    """
    CNN for SOH regression.
    Input  : (batch, 32, 9, 1) — cycles × features as 2D image
    Output : (batch, 1)        — SOH value
    """
    inp = Input(shape=(IMG_H, IMG_W, 1), name='cnn_input')

    # Block 1
    x = layers.Conv2D(32, (3,3), padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    # Block 2
    x = layers.Conv2D(64, (3,3), padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # Regression head
    x   = layers.Dense(64, activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, name='soh_output')(x)

    return Model(inp, out, name='CNN_BatteryBench')

# ── LSTM ──────────────────────────────────────────────────────
def build_lstm():
    """
    LSTM for SOH regression.
    Input  : (batch, 32, 9)  — 32 cycles × 9 features
    Output : (batch, 1)      — SOH value
    """
    inp = Input(shape=(SEQ_LEN, N_FEATURES), name='lstm_input')

    x = layers.LSTM(64, return_sequences=True,  name='lstm1')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.LSTM(32, return_sequences=False, name='lstm2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(1, name='soh_output')(x)

    return Model(inp, out, name='LSTM_BatteryBench')

# ── Positional Encoding ───────────────────────────────────────
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        positions = np.arange(max_len)[:, np.newaxis]
        dims      = np.arange(d_model)[np.newaxis, :]
        angles    = positions / np.power(
            10000, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pos_enc = tf.cast(
            angles[np.newaxis, :, :], tf.float32)

    def call(self, x):
        return x + self.pos_enc[:, :tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'max_len': self.max_len,
                    'd_model': self.d_model})
        return cfg

# ── Transformer block ─────────────────────────────────────────
def transformer_block(x, d_model, n_heads, d_ff, dropout=0.2):
    attn = layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=d_model // n_heads,
        dropout=dropout)(x, x)
    attn = layers.Dropout(dropout)(attn)
    x    = layers.LayerNormalization(epsilon=1e-6)(x + attn)
    ffn  = layers.Dense(d_ff, activation='relu')(x)
    ffn  = layers.Dropout(dropout)(ffn)
    ffn  = layers.Dense(d_model)(ffn)
    ffn  = layers.Dropout(dropout)(ffn)
    x    = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    return x

# ── Transformer ───────────────────────────────────────────────
def build_transformer():
    """
    Transformer encoder for SOH regression.
    Input  : (batch, 32, 9)  — 32 cycles × 9 features
    Output : (batch, 1)      — SOH value
    """
    inp = Input(shape=(SEQ_LEN, N_FEATURES), name='trans_input')

    x = layers.Dense(D_MODEL, name='input_projection')(inp)
    x = PositionalEncoding(SEQ_LEN, D_MODEL,
                           name='pos_encoding')(x)
    x = layers.Dropout(0.2)(x)

    for _ in range(N_BLOCKS):
        x = transformer_block(x, D_MODEL, N_HEADS, D_FF, 0.2)

    x = layers.GlobalAveragePooling1D(name='gap')(x)

    x   = layers.Dense(64, activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(1, name='soh_output')(x)

    return Model(inp, out, name='Transformer_BatteryBench')

# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    # MAPE — avoid division by zero
    mask = y_true > 1.0
    mape = float(np.mean(np.abs(
        (y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

# ══════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════

def plot_history(history, model_name):
    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
    train_mae  = history.history['mae']
    val_mae    = history.history['val_mae']
    epochs     = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'BatteryBench — {model_name} Training History\n'
                 f'XJTU Battery SOH Estimation',
                 fontsize=13, fontweight='bold')

    # Loss
    ax = axes[0]
    ax.plot(epochs, train_loss, color='#1a7abf',
            label='Train loss', linewidth=2)
    ax.plot(epochs, val_loss,   color='#27ae60',
            label='Val loss',   linewidth=2, linestyle='--')
    best_ep = int(np.argmin(val_loss)) + 1
    ax.scatter([best_ep], [min(val_loss)],
               color='#27ae60', s=80, zorder=5)
    ax.annotate(f'Best: {min(val_loss):.4f} (ep {best_ep})',
                xy=(best_ep, min(val_loss)),
                xytext=(8, 8), textcoords='offset points',
                fontsize=9, color='#27ae60')
    ax.set_title('MSE Loss', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    # MAE
    ax = axes[1]
    ax.plot(epochs, train_mae, color='#c0392b',
            label='Train MAE', linewidth=2)
    ax.plot(epochs, val_mae,   color='#e67e22',
            label='Val MAE',   linewidth=2, linestyle='--')
    best_mae_ep = int(np.argmin(val_mae)) + 1
    ax.scatter([best_mae_ep], [min(val_mae)],
               color='#e67e22', s=80, zorder=5)
    ax.annotate(f'Best: {min(val_mae):.4f} (ep {best_mae_ep})',
                xy=(best_mae_ep, min(val_mae)),
                xytext=(8, 8), textcoords='offset points',
                fontsize=9, color='#e67e22')
    ax.set_title('MAE (%SOH)', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MAE')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    name = model_name.lower().replace(' ', '_')
    path = os.path.join(RESULTS_DIR, f'{name}_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

def plot_predictions(y_true, y_pred, model_name, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'BatteryBench — {model_name} Predictions\n'
                 f'RMSE={metrics["rmse"]:.3f}%  '
                 f'MAE={metrics["mae"]:.3f}%  '
                 f'R²={metrics["r2"]:.4f}',
                 fontsize=13, fontweight='bold')

    # Scatter: predicted vs true
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.3, s=8,
               color='#1a7abf', label='Predictions')
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2,
            label='Perfect prediction')
    ax.set_xlabel('True SOH (%)')
    ax.set_ylabel('Predicted SOH (%)')
    ax.set_title('Predicted vs True SOH', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Residuals
    ax = axes[1]
    residuals = y_pred - y_true
    ax.hist(residuals, bins=50, color='#1a7abf',
            alpha=0.7, edgecolor='white')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(residuals.mean(), color='orange',
               linestyle='--', linewidth=1.5,
               label=f'Mean={residuals.mean():.3f}%')
    ax.set_xlabel('Residual (Predicted − True) %SOH')
    ax.set_ylabel('Count')
    ax.set_title('Residual Distribution', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    name = model_name.lower().replace(' ', '_')
    path = os.path.join(RESULTS_DIR, f'{name}_predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

# ══════════════════════════════════════════════════════════════
# MYSQL SAVE
# ══════════════════════════════════════════════════════════════

def save_to_mysql(model_name, model, metrics_val,
                  metrics_test, training_time,
                  epochs_trained, model_path, input_shape):
    try:
        import mysql.connector
        conn   = mysql.connector.connect(
            host=DB_HOST, port=DB_PORT,
            user=DB_USER, password=DB_PASSWORD,
            database=DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO model_results (
                model_name, input_type, input_shape, params,
                rmse, mae, mape, r2,
                test_rmse, test_mae, test_r2,
                training_time_sec, epochs_trained,
                batch_size, learning_rate, model_path
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            model_name, 'summary_features', input_shape,
            model.count_params(),
            round(metrics_val['rmse'],  4),
            round(metrics_val['mae'],   4),
            round(metrics_val['mape'],  4),
            round(metrics_val['r2'],    4),
            round(metrics_test['rmse'], 4),
            round(metrics_test['mae'],  4),
            round(metrics_test['r2'],   4),
            round(training_time, 1),
            epochs_trained,
            BATCH_SIZE, 1e-3, model_path
        ))
        conn.commit()
        cursor.close(); conn.close()
        print(f'  Results saved to MySQL ✅')
    except Exception as e:
        print(f'  MySQL save skipped: {e}')

# ══════════════════════════════════════════════════════════════
# TRAIN ONE MODEL
# ══════════════════════════════════════════════════════════════

def train_model(model, model_name, model_path,
                X_train, y_train, X_val, y_val,
                X_test, y_test, input_shape):

    print(f'\n{"="*55}')
    print(f'  Training {model_name}')
    print(f'  Params  : {model.count_params():,}')
    print(f'  Input   : {input_shape}')
    print(f'{"="*55}')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='mse',
        metrics=['mae']
    )

    callbacks = [
        ModelCheckpoint(model_path, monitor='val_mae',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=8, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=1),
    ]

    start   = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    training_time  = time.time() - start
    epochs_trained = len(history.history['loss'])

    print(f'\n  Training time : {training_time:.1f}s '
          f'({training_time/60:.1f} min)')
    print(f'  Epochs        : {epochs_trained}')

    # Evaluate on val and test
    print(f'\nEvaluating {model_name}...')
    y_val_pred  = model.predict(X_val,  batch_size=BATCH_SIZE,
                                verbose=0).flatten()
    y_test_pred = model.predict(X_test, batch_size=BATCH_SIZE,
                                verbose=0).flatten()

    metrics_val  = compute_metrics(y_val,  y_val_pred)
    metrics_test = compute_metrics(y_test, y_test_pred)

    print(f'\n── Val  metrics ─────────────────────────────────')
    print(f'  RMSE : {metrics_val["rmse"]:.4f}%')
    print(f'  MAE  : {metrics_val["mae"]:.4f}%')
    print(f'  MAPE : {metrics_val["mape"]:.2f}%')
    print(f'  R²   : {metrics_val["r2"]:.4f}')

    print(f'\n── Test metrics (RW protocol) ───────────────────')
    print(f'  RMSE : {metrics_test["rmse"]:.4f}%')
    print(f'  MAE  : {metrics_test["mae"]:.4f}%')
    print(f'  MAPE : {metrics_test["mape"]:.2f}%')
    print(f'  R²   : {metrics_test["r2"]:.4f}')

    # Plots
    print('\nSaving plots...')
    plot_history(history, model_name)
    plot_predictions(y_test, y_test_pred, model_name, metrics_test)

    # Save to MySQL
    save_to_mysql(model_name, model, metrics_val,
                  metrics_test, training_time,
                  epochs_trained, model_path, input_shape)

    return {
        'val_rmse'  : metrics_val['rmse'],
        'val_mae'   : metrics_val['mae'],
        'val_r2'    : metrics_val['r2'],
        'test_rmse' : metrics_test['rmse'],
        'test_mae'  : metrics_test['mae'],
        'test_r2'   : metrics_test['r2'],
        'time'      : training_time,
        'params'    : model.count_params(),
    }

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('\nBatteryBench — Unified Trainer')
    print('CNN · LSTM · Transformer · SOH Regression')
    print('=' * 55)
    setup_gpu()

    # ── Load data ─────────────────────────────────────────────
    print('\nLoading dataset...')
    splits, scaler, meta = load_dataset()

    X_seq_tr, y_tr  = splits['train']
    X_seq_val, y_val = splits['val']
    X_seq_te, y_te  = splits['test']

    X_img_tr  = to_cnn_input(X_seq_tr)
    X_img_val = to_cnn_input(X_seq_val)
    X_img_te  = to_cnn_input(X_seq_te)

    print(f'\n  Train : {X_seq_tr.shape}  y=[{y_tr.min():.1f}-{y_tr.max():.1f}%]')
    print(f'  Val   : {X_seq_val.shape}  y=[{y_val.min():.1f}-{y_val.max():.1f}%]')
    print(f'  Test  : {X_seq_te.shape}  y=[{y_te.min():.1f}-{y_te.max():.1f}%]')

    results = {}

    # ── CNN ───────────────────────────────────────────────────
    cnn_path = os.path.join(MODELS_DIR, 'cnn_best.h5')
    if os.path.exists(cnn_path):
        print('\n  CNN already trained — skipping')
        results['CNN'] = {}
    else:
        results['CNN'] = train_model(
            build_cnn(), 'CNN', cnn_path,
            X_img_tr, y_tr, X_img_val, y_val, X_img_te, y_te,
            f'(batch, {IMG_H}, {IMG_W}, 1)'
        )

    # ── LSTM ──────────────────────────────────────────────────
    lstm_path = os.path.join(MODELS_DIR, 'lstm_best.h5')
    if os.path.exists(lstm_path):
        print('\n  LSTM already trained — skipping')
        results['LSTM'] = {}
    else:
        results['LSTM'] = train_model(
            build_lstm(), 'LSTM', lstm_path,
            X_seq_tr, y_tr, X_seq_val, y_val, X_seq_te, y_te,
            f'(batch, {SEQ_LEN}, {N_FEATURES})'
        )

    # ── Transformer ───────────────────────────────────────────
    trans_path = os.path.join(MODELS_DIR, 'transformer_best.h5')
    if os.path.exists(trans_path):
        print('\n  Transformer already trained — skipping')
        results['Transformer'] = {}
    else:
        results['Transformer'] = train_model(
            build_transformer(), 'Transformer', trans_path,
            X_seq_tr, y_tr, X_seq_val, y_val, X_seq_te, y_te,
            f'(batch, {SEQ_LEN}, {N_FEATURES})'
        )

    # ── Final comparison ──────────────────────────────────────
    print('\n\n' + '='*65)
    print('BATTERYBENCH — FINAL RESULTS')
    print('SOH Regression · XJTU Battery · RW Protocol Test')
    print('='*65)
    print(f'{"Model":<14} {"ValRMSE":>9} {"ValMAE":>8}'
          f' {"TestRMSE":>10} {"TestMAE":>9}'
          f' {"TestR2":>8} {"Params":>10} {"Time(s)":>9}')
    print('-'*65)
    for name, r in results.items():
        if r:
            print(f'{name:<14} {r["val_rmse"]:>9.4f}'
                  f' {r["val_mae"]:>8.4f}'
                  f' {r["test_rmse"]:>10.4f}'
                  f' {r["test_mae"]:>9.4f}'
                  f' {r["test_r2"]:>8.4f}'
                  f' {r["params"]:>10,}'
                  f' {r["time"]:>9.1f}')

    # Save text report
    report = [
        '='*65,
        'BatteryBench — SOH Estimation Benchmark',
        'XJTU Battery Dataset · CNN vs LSTM vs Transformer',
        'Train: Batch-1 to 4 | Test: Batch-5 (RW protocol)',
        '='*65,
        f'{"Model":<14} {"ValRMSE":>9} {"TestRMSE":>10}'
        f' {"TestMAE":>9} {"TestR2":>8} {"Params":>10}',
        '-'*65,
    ]
    for name, r in results.items():
        if r:
            report.append(
                f'{name:<14} {r["val_rmse"]:>9.4f}'
                f' {r["test_rmse"]:>10.4f}'
                f' {r["test_mae"]:>9.4f}'
                f' {r["test_r2"]:>8.4f}'
                f' {r["params"]:>10,}')

    path = os.path.join(RESULTS_DIR, 'benchmark_report.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(report))
    print(f'\n  Report saved: {path}')
    print('\nBatteryBench training complete!\n')
