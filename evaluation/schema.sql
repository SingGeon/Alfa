-- Prediction evaluation log (see evaluation/storage.py).
--
-- All timestamps are UTC ISO-8601 strings in one fixed format
-- ("YYYY-MM-DDTHH:MM:SSZ"), so lexicographic order == chronological order
-- and date filters are plain string comparisons.
--
-- predicted_path: JSON list of {"timestamp", "price", "lower", "upper"},
--   one per forecast step; "timestamp" is the candle open time the model
--   forecast the close for (same convention as /api/predict).
-- actual_path: JSON list of {"timestamp", "price", "open", "high", "low"} -
--   the real candle ("price" = its close) for each of those timestamps,
--   appended as each one closes (open/high/low absent when only CoinGecko
--   had the price).
-- history_path: JSON list of {"timestamp", "open", "high", "low", "close"} -
--   the real candles the model saw right before predicting (drawn to the
--   left of the forecast in the visual snapshot).
-- target_time: timestamp of the LAST predicted candle.
-- resolved_at: when its real close becomes known = target_time + one
--   interval (that candle's close). Deterministic, so it also decides which
--   day's chart folder / daily overview a prediction belongs to.

CREATE TABLE IF NOT EXISTS predictions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at            TEXT    NOT NULL,
    interval              TEXT    NOT NULL,
    model_name            TEXT    NOT NULL,
    price_at_prediction   REAL    NOT NULL,
    horizon_steps         INTEGER NOT NULL,
    target_time           TEXT    NOT NULL,
    resolved_at           TEXT    NOT NULL,
    predicted_path        TEXT    NOT NULL,
    predicted_final_price REAL    NOT NULL,
    confidence            REAL,
    signal                TEXT CHECK (signal IN ('Cumpără', 'Vinde', 'Așteaptă')),
    actual_path           TEXT    NOT NULL DEFAULT '[]',
    history_path          TEXT    NOT NULL DEFAULT '[]',
    actual_final_price    REAL,
    abs_error             REAL,
    pct_error             REAL,
    direction_correct     INTEGER,             -- 0/1, NULL until completed
    status                TEXT    NOT NULL DEFAULT 'pending'
                          CHECK (status IN ('pending', 'completed', 'expired')),
    completed_at          TEXT,
    UNIQUE (interval, model_name, created_at)
);

CREATE INDEX IF NOT EXISTS idx_predictions_status  ON predictions (status, resolved_at);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions (created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_resolved ON predictions (resolved_at);
CREATE INDEX IF NOT EXISTS idx_predictions_model   ON predictions (model_name, interval);
