-- ============================================================
-- BatteryBench — MySQL Schema
-- XJTU Battery Dataset · SOH Estimation
-- Auto-runs on first docker compose up
-- ============================================================

CREATE DATABASE IF NOT EXISTS batterybench;
USE batterybench;

-- ── Batches ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS batches (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(20)  NOT NULL UNIQUE,
    charge_protocol VARCHAR(30)  NOT NULL,
    description     VARCHAR(200),
    n_batteries     INT          NOT NULL DEFAULT 0,
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

-- ── Batteries ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS batteries (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    batch_id        INT          NOT NULL,
    name            VARCHAR(30)  NOT NULL UNIQUE,
    filename        VARCHAR(50)  NOT NULL,
    cycle_life      INT          NOT NULL,
    initial_cap_ah  FLOAT,
    eol_cycle       INT,
    split           VARCHAR(10)  NOT NULL DEFAULT 'train',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id) REFERENCES batches(id),
    INDEX idx_batch (batch_id),
    INDEX idx_split (split)
);

-- ── Cycles ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cycles (
    id                       INT AUTO_INCREMENT PRIMARY KEY,
    battery_id               INT   NOT NULL,
    cycle_number             INT   NOT NULL,
    soh                      FLOAT NOT NULL,
    discharge_capacity_ah    FLOAT,
    charge_capacity_ah       FLOAT,
    discharge_power_wh       FLOAT,
    charge_power_wh          FLOAT,
    charge_median_voltage    FLOAT,
    discharge_median_voltage FLOAT,
    charge_mean_voltage      FLOAT,
    discharge_mean_voltage   FLOAT,
    is_eol                   BOOLEAN DEFAULT FALSE,
    created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (battery_id) REFERENCES batteries(id),
    INDEX idx_battery_cycle  (battery_id, cycle_number),
    INDEX idx_soh            (soh)
);

-- ── Model Results ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_results (
    id                 INT AUTO_INCREMENT PRIMARY KEY,
    model_name         VARCHAR(50)  NOT NULL,
    input_type         VARCHAR(50)  NOT NULL,
    input_shape        VARCHAR(100),
    params             INT,
    rmse               FLOAT,
    mae                FLOAT,
    mape               FLOAT,
    r2                 FLOAT,
    test_rmse          FLOAT,
    test_mae           FLOAT,
    test_r2            FLOAT,
    training_time_sec  FLOAT,
    epochs_trained     INT,
    batch_size         INT,
    learning_rate      FLOAT,
    model_path         VARCHAR(200),
    trained_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Views ─────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_battery_summary AS
SELECT
    ba.name                  AS batch,
    ba.charge_protocol,
    b.name                   AS battery,
    b.cycle_life,
    b.eol_cycle,
    b.split,
    ROUND(MIN(c.soh), 2)     AS min_soh,
    ROUND(MAX(c.soh), 2)     AS max_soh,
    ROUND(AVG(c.soh), 2)     AS avg_soh,
    COUNT(c.id)              AS total_cycles
FROM batteries b
JOIN batches ba ON ba.id = b.batch_id
JOIN cycles  c  ON c.battery_id = b.id
GROUP BY b.id
ORDER BY ba.name, b.name;

CREATE OR REPLACE VIEW v_model_comparison AS
SELECT
    model_name,
    input_type,
    params,
    ROUND(rmse,4)       AS rmse,
    ROUND(mae,4)        AS mae,
    ROUND(mape,2)       AS mape_pct,
    ROUND(r2,4)         AS r2,
    ROUND(test_rmse,4)  AS test_rmse,
    ROUND(test_mae,4)   AS test_mae,
    ROUND(test_r2,4)    AS test_r2,
    ROUND(training_time_sec/60,1) AS training_min,
    epochs_trained
FROM model_results
ORDER BY test_rmse ASC;

CREATE OR REPLACE VIEW v_soh_degradation AS
SELECT
    b.name              AS battery,
    ba.charge_protocol,
    ba.name             AS batch,
    b.split,
    c.cycle_number,
    ROUND(c.soh,3)      AS soh,
    c.discharge_capacity_ah,
    c.discharge_mean_voltage
FROM cycles c
JOIN batteries b  ON b.id = c.battery_id
JOIN batches   ba ON ba.id = b.batch_id
ORDER BY ba.name, b.name, c.cycle_number;

-- ── Seed batch metadata ───────────────────────────────────────
INSERT IGNORE INTO batches (name, charge_protocol, description, n_batteries) VALUES
('Batch-1', '2C',        '2C constant current charge · 8 batteries',  8),
('Batch-2', '3C',        '3C constant current charge · 15 batteries', 15),
('Batch-3', 'R2.5',      'R2.5 ohm resistive load · 8 batteries',     8),
('Batch-4', 'R3',        'R3 ohm resistive load · 8 batteries',       8),
('Batch-5', 'RW',        'Random walk charge protocol · 8 batteries', 8),
-- Batch-6 excluded: LEO satellite partial cycles
-- incompatible with SOH definition for EV/ground applications;
