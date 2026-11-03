-- Схема БД ExperimentLab. Скрипт безопасно запускать повторно.

CREATE TABLE IF NOT EXISTS experiment_groups (
    group_id   SMALLINT PRIMARY KEY,
    group_name TEXT NOT NULL UNIQUE,
    CONSTRAINT ck_group_name CHECK (group_name IN ('control', 'treatment'))
);

INSERT INTO experiment_groups (group_id, group_name)
VALUES (1, 'control'), (2, 'treatment')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS users (
    user_id            BIGINT PRIMARY KEY,
    group_id           SMALLINT NOT NULL REFERENCES experiment_groups (group_id),
    signup_ts          TIMESTAMP NOT NULL,
    device_type        TEXT NOT NULL,
    country            TEXT NOT NULL,
    traffic_source     TEXT NOT NULL,
    is_new_user        BOOLEAN NOT NULL,
    age                SMALLINT NOT NULL,
    sessions_prior_30d INTEGER NOT NULL,
    cart_items         INTEGER NOT NULL,
    cart_value         NUMERIC(12, 2) NOT NULL,
    CONSTRAINT ck_device CHECK (device_type IN ('mobile', 'desktop', 'tablet')),
    CONSTRAINT ck_source CHECK (
        traffic_source IN ('organic', 'paid_search', 'direct', 'email', 'social')),
    CONSTRAINT ck_country CHECK (country ~ '^[A-Z]{2,5}$'),
    CONSTRAINT ck_age CHECK (age BETWEEN 18 AND 75),
    CONSTRAINT ck_sessions CHECK (sessions_prior_30d >= 0),
    CONSTRAINT ck_cart_items CHECK (cart_items >= 1),
    CONSTRAINT ck_cart_value CHECK (cart_value > 0)
);
CREATE INDEX IF NOT EXISTS ix_users_group_id  ON users (group_id);
CREATE INDEX IF NOT EXISTS ix_users_signup_ts ON users (signup_ts);

CREATE TABLE IF NOT EXISTS user_outcomes (
    user_id     BIGINT PRIMARY KEY REFERENCES users (user_id) ON DELETE CASCADE,
    converted   BOOLEAN NOT NULL,
    revenue     NUMERIC(12, 2) NOT NULL DEFAULT 0,
    purchase_ts TIMESTAMP,
    CONSTRAINT ck_revenue_nonneg CHECK (revenue >= 0),
    CONSTRAINT ck_buyer_consistency CHECK (
        (converted AND revenue > 0 AND purchase_ts IS NOT NULL)
        OR (NOT converted AND revenue = 0 AND purchase_ts IS NULL))
);

CREATE TABLE IF NOT EXISTS load_log (
    load_id     SERIAL PRIMARY KEY,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    file_sha256 TEXT NOT NULL,
    rows_in_csv INTEGER,
    rows_in_db  INTEGER,
    status      TEXT NOT NULL,
    error       TEXT,
    CONSTRAINT ck_load_status CHECK (status IN ('running', 'success', 'failed'))
);
