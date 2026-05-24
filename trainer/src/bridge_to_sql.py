"""
BatteryBench — Bridge to MySQL
Populates batteries and cycles tables from XJTU .mat files

Tables populated:
    batteries : one row per battery (metadata + split)
    cycles    : one row per cycle   (SOH + summary features)

Run: python src/bridge_to_sql.py
"""

import os
import sys
import numpy as np
import scipy.io
import mysql.connector

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SRC_DIR  = os.path.join(BASE_DIR, 'src')
sys.path.insert(0, SRC_DIR)

# ── Database config ───────────────────────────────────────────
# Reads from environment variables (Docker) or uses defaults (WSL)
DB_HOST     = os.environ.get('DB_HOST',     'localhost')
DB_PORT     = int(os.environ.get('DB_PORT', '3307'))
DB_USER     = os.environ.get('DB_USER',     'battuser')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'battery123')
DB_NAME     = os.environ.get('DB_NAME',     'batterybench')

# ── Batch config ──────────────────────────────────────────────
BATCH_CONFIG = {
    'Batch-1': {'protocol': '2C',   'split': 'trainval'},
    'Batch-2': {'protocol': '3C',   'split': 'trainval'},
    'Batch-3': {'protocol': 'R2.5', 'split': 'trainval'},
    'Batch-4': {'protocol': 'R3',   'split': 'trainval'},
    'Batch-5': {'protocol': 'RW',   'split': 'test'},
    # Batch-6 excluded: LEO satellite partial cycles
}

SKIP_FILES  = ['Temperature_Compensation_Data.mat']
VAL_RATIO   = 0.20
EOL_SOH     = 80.0

# ── Connect to MySQL ──────────────────────────────────────────
def get_conn():
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASSWORD,
        database=DB_NAME
    )

# ── Get batch_id ──────────────────────────────────────────────
def get_batch_id(cursor, batch_name):
    cursor.execute(
        'SELECT id FROM batches WHERE name=%s', (batch_name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    raise ValueError(f'Batch {batch_name} not found in DB')

# ── Insert one battery ────────────────────────────────────────
def insert_battery(cursor, batch_id, name, filename,
                   cycle_life, initial_cap, eol_cycle,
                   split, final_soh):
    cursor.execute("""
        INSERT IGNORE INTO batteries
            (batch_id, name, filename, cycle_life,
             initial_cap_ah, eol_cycle, split)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (batch_id, name, filename, cycle_life,
          initial_cap, eol_cycle, split))
    cursor.execute(
        'SELECT id FROM batteries WHERE name=%s', (name,))
    return cursor.fetchone()[0]

# ── Insert cycles for one battery ─────────────────────────────
def insert_cycles(cursor, battery_id, summary, soh, n):
    rows = []
    for i in range(n):
        is_eol = bool(soh[i] < EOL_SOH)
        rows.append((
            battery_id,
            i + 1,                                          # cycle_number
            round(float(soh[i]), 4),                        # soh
            round(float(summary['discharge_capacity_Ah'][i]), 4),
            round(float(summary['charge_capacity_Ah'][i]),  4),
            round(float(summary['discharge_power_Wh'][i]),  4),
            round(float(summary['charge_power_Wh'][i]),     4),
            round(float(summary['charge_median_voltage'][i]),    4),
            round(float(summary['discharge_median_voltage'][i]), 4),
            round(float(summary['charge_mean_voltage'][i]),      4),
            round(float(summary['discharge_mean_voltage'][i]),   4),
            is_eol,
        ))

    cursor.executemany("""
        INSERT IGNORE INTO cycles
            (battery_id, cycle_number, soh,
             discharge_capacity_ah, charge_capacity_ah,
             discharge_power_wh,    charge_power_wh,
             charge_median_voltage, discharge_median_voltage,
             charge_mean_voltage,   discharge_mean_voltage,
             is_eol)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, rows)

# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\nBatteryBench — Bridge to MySQL')
    print('=' * 55)
    print(f'  Host     : {DB_HOST}:{DB_PORT}')
    print(f'  Database : {DB_NAME}')

    # Connect
    try:
        conn   = get_conn()
        cursor = conn.cursor()
        print('  Connected ✅\n')
    except Exception as e:
        print(f'  Connection failed: {e}')
        print(f'  Is MySQL running? docker compose up -d mysql')
        sys.exit(1)

    total_batteries = 0
    total_cycles    = 0

    print(f'  {"Batch":<10} {"Battery":<30} {"Cycles":>7}'
          f' {"EOL":>6} {"Split"}')
    print('  ' + '-'*65)

    for batch_name, cfg in BATCH_CONFIG.items():
        batch_path = os.path.join(DATA_DIR, batch_name)
        if not os.path.exists(batch_path):
            print(f'  WARNING: {batch_path} not found — skipping')
            continue

        batch_id = get_batch_id(cursor, batch_name)
        files    = sorted([f for f in os.listdir(batch_path)
                           if f.endswith('.mat')
                           and f not in SKIP_FILES])

        for fname in files:
            path = os.path.join(batch_path, fname)
            mat  = scipy.io.loadmat(path, simplify_cells=True)
            keys = [k for k in mat.keys() if not k.startswith('__')]
            sk   = next((k for k in keys if 'sum' in k.lower()), None)
            if not sk:
                continue

            s = mat[sk]
            n = int(s['cycle_life'])

            # SOH calculation
            cap     = np.array(s['discharge_capacity_Ah'],
                               dtype=np.float32)
            ref_cap = float(np.max(cap[:min(5, len(cap))]))
            soh     = np.clip(cap / ref_cap * 100.0,
                              0.0, 100.0).astype(np.float32)
            eol     = next((i+1 for i, v in enumerate(soh)
                            if v < EOL_SOH), n)

            # Determine split: last 20% of trainval -> val
            if cfg['split'] == 'trainval':
                split_idx = int(n * (1 - VAL_RATIO))
                # Store as 'train' or 'val' per battery
                # (split happens at cycle level in data loader)
                db_split = 'trainval'
            else:
                db_split = cfg['split']

            bname = fname.replace('.mat', '')

            # Insert battery
            battery_id = insert_battery(
                cursor, batch_id, bname, fname,
                n, float(cap[0]), eol, db_split,
                float(soh[-1])
            )

            # Insert cycles
            insert_cycles(cursor, battery_id, s, soh, n)
            conn.commit()

            total_batteries += 1
            total_cycles    += n

            print(f'  {batch_name:<10} {bname:<30}'
                  f' {n:>7} {eol:>6}  {db_split}')

    # Summary
    print('\n── Database Summary ─────────────────────────────────')
    cursor.execute('SELECT COUNT(*) FROM batteries')
    print(f'  Batteries : {cursor.fetchone()[0]}')
    cursor.execute('SELECT COUNT(*) FROM cycles')
    print(f'  Cycles    : {cursor.fetchone()[0]:,}')

    # SOH distribution
    cursor.execute("""
        SELECT
            ROUND(AVG(soh), 2)  AS avg_soh,
            ROUND(MIN(soh), 2)  AS min_soh,
            ROUND(MAX(soh), 2)  AS max_soh
        FROM cycles
    """)
    row = cursor.fetchone()
    print(f'  SOH range : {row[1]}% – {row[2]}%  mean={row[0]}%')

    # Per batch
    print('\n── Per Batch ─────────────────────────────────────────')
    cursor.execute("""
        SELECT ba.name, ba.charge_protocol,
               COUNT(DISTINCT b.id)  AS batteries,
               COUNT(c.id)           AS cycles,
               ROUND(AVG(c.soh), 1)  AS avg_soh
        FROM batches ba
        JOIN batteries b ON b.batch_id = ba.id
        JOIN cycles    c ON c.battery_id = b.id
        GROUP BY ba.id
        ORDER BY ba.name
    """)
    print(f'  {"Batch":<10} {"Protocol":<10}'
          f' {"Batteries":>10} {"Cycles":>8} {"AvgSOH":>8}')
    print('  ' + '-'*50)
    for r in cursor.fetchall():
        print(f'  {r[0]:<10} {r[1]:<10}'
              f' {r[2]:>10} {r[3]:>8,} {r[4]:>7.1f}%')

    cursor.close()
    conn.close()
    print(f'\n  Total batteries : {total_batteries}')
    print(f'  Total cycles    : {total_cycles:,}')
    print('\nBatteryBench database populated! ✅\n')
