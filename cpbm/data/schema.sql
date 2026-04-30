CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE individuals (
    ind_id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    geo_code        VARCHAR(20) NOT NULL,
    stratum         SMALLINT    CHECK (stratum BETWEEN 1 AND 5),
    age_band        VARCHAR(10),
    income_proxy    NUMERIC(12,2),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE products (
    product_id      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    category        VARCHAR(100) NOT NULL,
    sub_category    VARCHAR(100),
    price_tier      SMALLINT    CHECK (price_tier BETWEEN 1 AND 3),
    brand           VARCHAR(100),
    base_price      NUMERIC(10,2)
);

CREATE TABLE transactions (
    txn_id          UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    ind_id          UUID          REFERENCES individuals(ind_id) ON DELETE CASCADE,
    product_id      UUID          REFERENCES products(product_id),
    amount          NUMERIC(10,2) NOT NULL CHECK (amount > 0),
    quantity        SMALLINT      DEFAULT 1,
    txn_timestamp   TIMESTAMPTZ   NOT NULL,
    channel         VARCHAR(20)   CHECK (channel IN ('retail','online','wholesale')),
    promo_flag      BOOLEAN       DEFAULT FALSE,
    geo_code        VARCHAR(20)
);

CREATE TABLE pulse_signatures (
    sig_id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    ind_id          UUID        REFERENCES individuals(ind_id),
    period_start    DATE        NOT NULL,
    period_end      DATE        NOT NULL,
    f_score         NUMERIC(8,4),
    s_score         NUMERIC(6,4),
    sigma_timing    NUMERIC(8,4),
    e_low           NUMERIC(8,4),
    e_high          NUMERIC(8,4),
    c_max           NUMERIC(8,4),
    h_entropy       NUMERIC(6,4),
    comp_vector     JSONB,
    cluster_label   SMALLINT,
    UNIQUE (ind_id, period_start)
);

CREATE TABLE social_network (
    edge_id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    node_i          UUID        REFERENCES individuals(ind_id),
    node_j          UUID        REFERENCES individuals(ind_id),
    strength        NUMERIC(4,3) CHECK (strength BETWEEN 0 AND 1),
    relation_type   VARCHAR(30),
    CHECK (node_i < node_j)
);

CREATE TABLE npi_scores (
    npi_id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    geo_code        VARCHAR(20) NOT NULL,
    category        VARCHAR(100),
    period          DATE        NOT NULL,
    M_value         NUMERIC(15,2),
    V_value         NUMERIC(10,4),
    R_value         NUMERIC(6,4),
    E_value         NUMERIC(10,4),
    npi_value       NUMERIC(12,4),
    tau_days        NUMERIC(8,2),
    UNIQUE (geo_code, category, period)
);

CREATE TABLE market_signals (
    signal_id       UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    geo_code        VARCHAR(20),
    signal_type     VARCHAR(50),
    value           NUMERIC(15,4),
    signal_date     DATE        NOT NULL,
    source          VARCHAR(100)
);

CREATE INDEX idx_txn_ind_time    ON transactions (ind_id, txn_timestamp DESC);
CREATE INDEX idx_txn_prod_time   ON transactions (product_id, txn_timestamp DESC);
CREATE INDEX idx_sig_period      ON pulse_signatures (period_start, period_end);
CREATE INDEX idx_npi_geo_period  ON npi_scores (geo_code, period DESC);
CREATE INDEX idx_signals_geo     ON market_signals (geo_code, signal_date DESC);