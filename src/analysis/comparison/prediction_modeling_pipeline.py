"""Cross-platform data profiling and first-pass predictive modeling pipeline."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.analysis import Analysis, AnalysisOutput

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class DatasetProfile:
    dataset: str
    row_count: int
    time_column: str | None
    min_time: str | None
    max_time: str | None
    id_column: str | None
    unique_ids: int | None
    missingness: dict[str, float]
    columns: list[dict[str, str]]


class PredictionModelingPipelineAnalysis(Analysis):
    """Profiles Parquet data and runs leakage-aware baseline predictive models."""

    def __init__(self):
        super().__init__(
            name="prediction_modeling_pipeline",
            description="Data profiling + leakage-safe first-pass predictive models",
        )
        self.base_dir = Path(__file__).parent.parent.parent.parent
        self.output_dir = self.base_dir / "output"
        self.data_dir = self.base_dir / "data"
        self.kalshi_markets = self.data_dir / "kalshi" / "markets"
        self.kalshi_trades = self.data_dir / "kalshi" / "trades"
        self.polymarket_markets = self.data_dir / "polymarket" / "markets"
        self.polymarket_trades = self.data_dir / "polymarket" / "trades"
        self.polymarket_blocks = self.data_dir / "polymarket" / "blocks"
        self.outcome_horizons = [24, 6, 1]
        self.movement_horizons = [1, 6]

    def run(self) -> AnalysisOutput:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        kalshi_profile = self._profile_platform(
            "kalshi",
            {
                "markets": self.kalshi_markets,
                "trades": self.kalshi_trades,
            },
        )
        polymarket_profile = self._profile_platform(
            "polymarket",
            {
                "markets": self.polymarket_markets,
                "trades": self.polymarket_trades,
                "blocks": self.polymarket_blocks,
            },
        )

        metrics_rows: list[dict[str, object]] = []
        ablation_rows: list[dict[str, object]] = []

        if self._has_parquet(self.kalshi_markets) and self._has_parquet(self.kalshi_trades):
            with self.progress("Running Kalshi predictive modeling"):
                outcome_df = self._build_kalshi_outcome_dataset()
                outcome_metrics, outcome_ablation = self._run_outcome_models(outcome_df)
                metrics_rows.extend(outcome_metrics)
                ablation_rows.extend(outcome_ablation)

                movement_df = self._build_kalshi_movement_dataset()
                movement_metrics, movement_ablation = self._run_movement_models(movement_df)
                metrics_rows.extend(movement_metrics)
                ablation_rows.extend(movement_ablation)

                self._plot_basic_distributions()
                self._plot_time_series_examples()
        else:
            self._write_missing_data_summary()

        metrics_df = pd.DataFrame(metrics_rows)
        ablation_df = pd.DataFrame(ablation_rows)

        metrics_path = self.output_dir / "prediction_modeling_metrics.csv"
        ablation_path = self.output_dir / "prediction_modeling_ablation.csv"
        if not metrics_df.empty:
            metrics_df.to_csv(metrics_path, index=False)
        else:
            pd.DataFrame(
                [
                    {
                        "platform": "kalshi",
                        "target": "none",
                        "horizon_hours": None,
                        "model": "not_run",
                        "metric": "reason",
                        "value": "Data missing or insufficient rows.",
                    }
                ]
            ).to_csv(metrics_path, index=False)

        if not ablation_df.empty:
            ablation_df.to_csv(ablation_path, index=False)
        else:
            pd.DataFrame(
                [
                    {
                        "platform": "kalshi",
                        "target": "none",
                        "horizon_hours": None,
                        "variant": "not_run",
                        "metric": "reason",
                        "value": "Data missing or insufficient rows.",
                    }
                ]
            ).to_csv(ablation_path, index=False)

        self._write_summary_md(kalshi_profile, polymarket_profile, metrics_df)

        fig = plt.figure(figsize=(11, 4))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.01,
            0.95,
            "Prediction Modeling Pipeline Complete",
            fontsize=14,
            weight="bold",
            va="top",
        )
        ax.text(
            0.01,
            0.68,
            f"Kalshi datasets profiled: {', '.join(kalshi_profile.get('datasets_available', [])) or 'none'}",
            fontsize=10,
        )
        ax.text(
            0.01,
            0.50,
            f"Polymarket datasets profiled: {', '.join(polymarket_profile.get('datasets_available', [])) or 'none'}",
            fontsize=10,
        )
        ax.text(
            0.01,
            0.32,
            f"Metrics rows written: {len(metrics_df)} | Ablation rows written: {len(ablation_df)}",
            fontsize=10,
        )
        ax.text(
            0.01,
            0.12,
            "See output/summary.md and output/*.csv/*.json/*.png for details.",
            fontsize=10,
        )
        fig.tight_layout()

        return AnalysisOutput(figure=fig, data=metrics_df if not metrics_df.empty else pd.DataFrame())

    def _profile_platform(self, platform: str, datasets: dict[str, Path]) -> dict[str, object]:
        con = duckdb.connect()
        profile: dict[str, object] = {
            "platform": platform,
            "datasets": {},
            "datasets_available": [],
            "field_semantics": self._field_semantics(platform),
        }
        csv_rows: list[dict[str, object]] = []

        for dataset_name, dataset_dir in datasets.items():
            if not self._has_parquet(dataset_dir):
                profile["datasets"][dataset_name] = {
                    "available": False,
                    "path": str(dataset_dir),
                    "reason": "No parquet files found",
                }
                csv_rows.append(
                    {
                        "platform": platform,
                        "dataset": dataset_name,
                        "metric": "available",
                        "value": False,
                    }
                )
                continue

            glob_path = str(dataset_dir / "*.parquet")
            info = self._dataset_profile(con, dataset_name, glob_path)
            profile["datasets"][dataset_name] = {
                "available": True,
                "path": str(dataset_dir),
                "row_count": info.row_count,
                "time_column": info.time_column,
                "time_range": {
                    "min": info.min_time,
                    "max": info.max_time,
                },
                "id_column": info.id_column,
                "unique_ids": info.unique_ids,
                "missingness": info.missingness,
                "columns": info.columns,
            }
            profile["datasets_available"].append(dataset_name)

            csv_rows.extend(
                [
                    {
                        "platform": platform,
                        "dataset": dataset_name,
                        "metric": "available",
                        "value": True,
                    },
                    {
                        "platform": platform,
                        "dataset": dataset_name,
                        "metric": "row_count",
                        "value": info.row_count,
                    },
                    {
                        "platform": platform,
                        "dataset": dataset_name,
                        "metric": "time_column",
                        "value": info.time_column,
                    },
                    {
                        "platform": platform,
                        "dataset": dataset_name,
                        "metric": "min_time",
                        "value": info.min_time,
                    },
                    {
                        "platform": platform,
                        "dataset": dataset_name,
                        "metric": "max_time",
                        "value": info.max_time,
                    },
                    {
                        "platform": platform,
                        "dataset": dataset_name,
                        "metric": "id_column",
                        "value": info.id_column,
                    },
                    {
                        "platform": platform,
                        "dataset": dataset_name,
                        "metric": "unique_ids",
                        "value": info.unique_ids,
                    },
                ]
            )
            for col, rate in info.missingness.items():
                csv_rows.append(
                    {
                        "platform": platform,
                        "dataset": dataset_name,
                        "metric": f"missing_rate__{col}",
                        "value": round(rate, 6),
                    }
                )

        profile_json = self.output_dir / f"data_profile_{platform}.json"
        profile_csv = self.output_dir / f"data_profile_{platform}.csv"
        profile_json.write_text(json.dumps(profile, indent=2))
        pd.DataFrame(csv_rows).to_csv(profile_csv, index=False)

        return profile

    def _dataset_profile(self, con: duckdb.DuckDBPyConnection, dataset_name: str, glob_path: str) -> DatasetProfile:
        desc = con.execute(f"DESCRIBE SELECT * FROM '{glob_path}'").df()
        columns = [{"name": row["column_name"], "type": row["column_type"]} for _, row in desc.iterrows()]
        column_names = [c["name"] for c in columns]

        row_count = int(con.execute(f"SELECT COUNT(*) FROM '{glob_path}'").fetchone()[0])
        time_column = self._pick_first(column_names, ["created_time", "close_time", "open_time", "end_date", "created_at", "timestamp", "_fetched_at"])
        id_column = self._pick_first(column_names, ["ticker", "id", "condition_id", "trade_id", "transaction_hash", "block_number"])

        min_time = None
        max_time = None
        if time_column:
            min_time, max_time = con.execute(
                f"SELECT CAST(MIN({self._qid(time_column)}) AS VARCHAR), CAST(MAX({self._qid(time_column)}) AS VARCHAR) FROM '{glob_path}'"
            ).fetchone()

        unique_ids = None
        if id_column:
            unique_ids = int(con.execute(f"SELECT COUNT(DISTINCT {self._qid(id_column)}) FROM '{glob_path}'").fetchone()[0])

        missing_cols = column_names[: min(18, len(column_names))]
        if row_count > 0 and missing_cols:
            missing_expr = ", ".join(
                [
                    f"AVG(CASE WHEN {self._qid(c)} IS NULL THEN 1.0 ELSE 0.0 END) AS {self._qid(c)}"
                    for c in missing_cols
                ]
            )
            miss_row = con.execute(f"SELECT {missing_expr} FROM '{glob_path}'").fetchone()
            missingness = {missing_cols[i]: float(miss_row[i] or 0.0) for i in range(len(missing_cols))}
        else:
            missingness = {}

        # Add platform-relevant entity cardinalities when columns exist.
        if dataset_name == "trades":
            for entity_col in ["ticker", "id", "maker", "taker", "trader"]:
                if entity_col in column_names:
                    val = int(con.execute(f"SELECT COUNT(DISTINCT {self._qid(entity_col)}) FROM '{glob_path}'").fetchone()[0])
                    missingness[f"distinct_{entity_col}"] = float(val)

        return DatasetProfile(
            dataset=dataset_name,
            row_count=row_count,
            time_column=time_column,
            min_time=min_time,
            max_time=max_time,
            id_column=id_column,
            unique_ids=unique_ids,
            missingness=missingness,
            columns=columns,
        )

    def _build_kalshi_outcome_dataset(self) -> pd.DataFrame:
        con = duckdb.connect()
        trades_glob = str(self.kalshi_trades / "*.parquet")
        markets_glob = str(self.kalshi_markets / "*.parquet")

        frames = []
        for horizon in self.outcome_horizons:
            df = con.execute(
                f"""
                WITH latest_markets AS (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY ticker
                               ORDER BY COALESCE(_fetched_at, close_time, created_time) DESC
                           ) AS rn
                    FROM '{markets_glob}'
                ),
                resolved AS (
                    SELECT
                        ticker,
                        result,
                        close_time,
                        close_time - INTERVAL '{horizon} hour' AS snapshot_time
                    FROM latest_markets
                    WHERE rn = 1
                      AND result IN ('yes', 'no')
                      AND close_time IS NOT NULL
                )
                SELECT
                    r.ticker,
                    r.close_time,
                    {horizon} AS horizon_hours,
                    CASE WHEN r.result = 'yes' THEN 1 ELSE 0 END AS target_yes,
                    arg_max(t.yes_price, t.created_time) AS last_yes_price,
                    arg_max(t.yes_price, t.created_time) FILTER (
                        WHERE t.created_time <= r.snapshot_time - INTERVAL '1 hour'
                    ) AS yes_price_1h_ago,
                    stddev_samp(t.yes_price) FILTER (
                        WHERE t.created_time > r.snapshot_time - INTERVAL '24 hour'
                    ) AS realized_vol_24h,
                    SUM(t.count) FILTER (
                        WHERE t.created_time > r.snapshot_time - INTERVAL '24 hour'
                    ) AS volume_24h,
                    COUNT(*) FILTER (
                        WHERE t.created_time > r.snapshot_time - INTERVAL '24 hour'
                    ) AS trade_count_24h,
                    SUM(t.count) AS volume_total,
                    COUNT(*) AS trade_count_total,
                    SUM(CASE WHEN t.taker_side = 'yes' THEN t.count ELSE -t.count END) AS net_yes_volume
                FROM resolved r
                LEFT JOIN '{trades_glob}' t
                  ON t.ticker = r.ticker
                 AND t.created_time <= r.snapshot_time
                GROUP BY 1, 2, 3, 4
                HAVING COUNT(t.created_time) > 0
                """
            ).df()
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        outcome_df = pd.concat(frames, ignore_index=True)
        outcome_df["close_time"] = pd.to_datetime(outcome_df["close_time"])
        outcome_df["last_yes_price"] = outcome_df["last_yes_price"].astype(float)
        outcome_df["yes_price_1h_ago"] = outcome_df["yes_price_1h_ago"].fillna(outcome_df["last_yes_price"])
        outcome_df["realized_vol_24h"] = outcome_df["realized_vol_24h"].fillna(0.0)
        outcome_df["volume_24h"] = outcome_df["volume_24h"].fillna(0.0)
        outcome_df["trade_count_24h"] = outcome_df["trade_count_24h"].fillna(0.0)
        outcome_df["volume_total"] = outcome_df["volume_total"].fillna(0.0)
        outcome_df["trade_count_total"] = outcome_df["trade_count_total"].fillna(0.0)
        outcome_df["net_yes_volume"] = outcome_df["net_yes_volume"].fillna(0.0)

        outcome_df["last_prob"] = (outcome_df["last_yes_price"] / 100.0).clip(0.01, 0.99)
        outcome_df["ret_1h"] = (outcome_df["last_yes_price"] - outcome_df["yes_price_1h_ago"]) / 100.0
        outcome_df["ofi"] = np.where(
            outcome_df["volume_total"] > 0,
            outcome_df["net_yes_volume"] / outcome_df["volume_total"],
            0.0,
        )
        outcome_df["time_to_expiry_hours"] = outcome_df["horizon_hours"].astype(float)

        return outcome_df

    def _run_outcome_models(self, outcome_df: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        if outcome_df.empty or outcome_df["ticker"].nunique() < 50:
            return [], []

        split = self._split_market_time(outcome_df, market_col="ticker", time_col="close_time", train_frac=0.8)
        train_df, test_df = split
        if train_df.empty or test_df.empty:
            return [], []

        feature_cols = [
            "last_prob",
            "ret_1h",
            "realized_vol_24h",
            "volume_24h",
            "trade_count_24h",
            "volume_total",
            "trade_count_total",
            "ofi",
            "time_to_expiry_hours",
        ]

        X_train = train_df[feature_cols]
        y_train = train_df["target_yes"].astype(int)
        X_test = test_df[feature_cols]
        y_test = test_df["target_yes"].astype(int)

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            return [], []

        baseline_p = test_df["last_prob"].astype(float).clip(1e-4, 1 - 1e-4).to_numpy()

        logistic = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
            ]
        )
        logistic.fit(X_train, y_train)
        logistic_p = logistic.predict_proba(X_test)[:, 1].clip(1e-4, 1 - 1e-4)

        gbm = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=4,
                        learning_rate=0.05,
                        max_iter=300,
                        random_state=42,
                    ),
                ),
            ]
        )
        gbm.fit(X_train, y_train)
        gbm_p = gbm.predict_proba(X_test)[:, 1].clip(1e-4, 1 - 1e-4)

        calibrated = CalibratedClassifierCV(estimator=gbm, method="isotonic", cv=3)
        calibrated.fit(X_train, y_train)
        cal_p = calibrated.predict_proba(X_test)[:, 1].clip(1e-4, 1 - 1e-4)

        metrics_rows: list[dict[str, object]] = []
        for horizon in sorted(test_df["horizon_hours"].unique()):
            idx = test_df["horizon_hours"] == horizon
            if idx.sum() < 20:
                continue
            yt = y_test[idx]
            for model_name, prob in [
                ("baseline_last_price", baseline_p[idx]),
                ("logistic", logistic_p[idx]),
                ("hist_gradient_boosting", gbm_p[idx]),
                ("hist_gradient_boosting_isotonic", cal_p[idx]),
            ]:
                metrics_rows.extend(
                    self._classification_metrics(
                        y_true=yt,
                        y_prob=prob,
                        platform="kalshi",
                        target="outcome_yes",
                        horizon_hours=int(horizon),
                        model=model_name,
                    )
                )

        full_metrics = []
        for model_name, prob in [
            ("baseline_last_price", baseline_p),
            ("logistic", logistic_p),
            ("hist_gradient_boosting", gbm_p),
            ("hist_gradient_boosting_isotonic", cal_p),
        ]:
            full_metrics.extend(
                self._classification_metrics(
                    y_true=y_test,
                    y_prob=prob,
                    platform="kalshi",
                    target="outcome_yes",
                    horizon_hours="all",
                    model=model_name,
                )
            )
        metrics_rows.extend(full_metrics)

        ablation_rows: list[dict[str, object]] = []
        X_train_price = train_df[["last_prob"]]
        X_test_price = test_df[["last_prob"]]
        logistic_price = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
            ]
        )
        logistic_price.fit(X_train_price, y_train)
        logistic_price_p = logistic_price.predict_proba(X_test_price)[:, 1].clip(1e-4, 1 - 1e-4)

        for variant, prob in [
            ("baseline_last_price", baseline_p),
            ("logistic_price_only", logistic_price_p),
            ("logistic_full", logistic_p),
            ("hgb_full", gbm_p),
        ]:
            ablation_rows.extend(
                self._ablation_from_classification(
                    y_true=y_test,
                    y_prob=prob,
                    platform="kalshi",
                    target="outcome_yes",
                    horizon_hours="all",
                    variant=variant,
                )
            )

        self._plot_calibration_curve(y_test, baseline_p, logistic_p, gbm_p, cal_p)
        self._plot_roc_curve(y_test, baseline_p, logistic_p, gbm_p, cal_p)
        self._plot_outcome_feature_importance(gbm, X_test, y_test, feature_cols)

        return metrics_rows, ablation_rows

    def _build_kalshi_movement_dataset(self) -> pd.DataFrame:
        con = duckdb.connect()
        trades_glob = str(self.kalshi_trades / "*.parquet")
        markets_glob = str(self.kalshi_markets / "*.parquet")

        trades_df = con.execute(
            f"""
            WITH latest_markets AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker
                           ORDER BY COALESCE(_fetched_at, close_time, created_time) DESC
                       ) AS rn
                FROM '{markets_glob}'
            ),
            resolved AS (
                SELECT ticker, close_time
                FROM latest_markets
                WHERE rn = 1
                  AND result IN ('yes', 'no')
                  AND close_time IS NOT NULL
            ),
            top_markets AS (
                SELECT t.ticker, COUNT(*) AS n_trades
                FROM '{trades_glob}' t
                INNER JOIN resolved r ON t.ticker = r.ticker
                WHERE t.created_time <= r.close_time
                GROUP BY 1
                ORDER BY n_trades DESC
                LIMIT 500
            )
            SELECT
                t.ticker,
                t.created_time,
                t.yes_price,
                t.count,
                t.taker_side
            FROM '{trades_glob}' t
            INNER JOIN top_markets m ON t.ticker = m.ticker
            WHERE t.yes_price BETWEEN 1 AND 99
            """
        ).df()

        if trades_df.empty:
            return pd.DataFrame()

        trades_df["created_time"] = pd.to_datetime(trades_df["created_time"])
        trades_df["yes_price"] = trades_df["yes_price"].astype(float) / 100.0
        trades_df = trades_df.sort_values(["ticker", "created_time"]).reset_index(drop=True)

        rows = []
        for ticker, group in trades_df.groupby("ticker", sort=False):
            g = group.copy()
            g = g.dropna(subset=["created_time", "yes_price"])
            if len(g) < 30:
                continue

            g["signed_count"] = np.where(g["taker_side"].eq("yes"), g["count"], -g["count"]).astype(float)
            g = g.set_index("created_time")
            g = g.sort_index()

            g["ret_1"] = g["yes_price"].diff().fillna(0.0)
            g["rolling_vol_1h"] = g["yes_price"].rolling("1h").std().fillna(0.0)
            g["rolling_volume_1h"] = g["count"].rolling("1h").sum().fillna(0.0)
            g["rolling_trade_intensity_1h"] = g["count"].rolling("1h").count().fillna(0.0)
            denom = g["rolling_volume_1h"].replace(0, np.nan)
            g["rolling_ofi_1h"] = (g["signed_count"].rolling("1h").sum() / denom).fillna(0.0)

            times = g.index.to_numpy()
            prices = g["yes_price"].to_numpy()

            for horizon in self.movement_horizons:
                future_times = g.index + pd.Timedelta(hours=horizon)
                pos = np.searchsorted(times, future_times.to_numpy(), side="left")
                valid = pos < len(g)
                if valid.sum() < 25:
                    continue

                curr = g.iloc[np.where(valid)[0]].copy()
                future_price = prices[pos[valid]]
                curr["future_return"] = future_price - curr["yes_price"].to_numpy()
                curr["future_up"] = (curr["future_return"] > 0).astype(int)
                curr["horizon_hours"] = horizon
                curr["ticker"] = ticker
                curr = curr.reset_index().rename(columns={"index": "created_time"})
                rows.append(
                    curr[
                        [
                            "ticker",
                            "created_time",
                            "horizon_hours",
                            "yes_price",
                            "ret_1",
                            "rolling_vol_1h",
                            "rolling_volume_1h",
                            "rolling_trade_intensity_1h",
                            "rolling_ofi_1h",
                            "future_return",
                            "future_up",
                        ]
                    ]
                )

        if not rows:
            return pd.DataFrame()

        movement_df = pd.concat(rows, ignore_index=True)
        movement_df = movement_df.sort_values("created_time").reset_index(drop=True)

        return movement_df

    def _run_movement_models(self, movement_df: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        if movement_df.empty or movement_df.shape[0] < 500:
            return [], []

        feature_cols = [
            "yes_price",
            "ret_1",
            "rolling_vol_1h",
            "rolling_volume_1h",
            "rolling_trade_intensity_1h",
            "rolling_ofi_1h",
        ]

        split_time = movement_df["created_time"].quantile(0.8)
        train_df = movement_df[movement_df["created_time"] <= split_time].copy()
        test_df = movement_df[movement_df["created_time"] > split_time].copy()
        if train_df.empty or test_df.empty:
            return [], []

        metrics_rows: list[dict[str, object]] = []
        ablation_rows: list[dict[str, object]] = []

        for horizon in sorted(movement_df["horizon_hours"].unique()):
            tr = train_df[train_df["horizon_hours"] == horizon]
            te = test_df[test_df["horizon_hours"] == horizon]
            if len(tr) < 100 or len(te) < 100:
                continue

            X_train = tr[feature_cols]
            X_test = te[feature_cols]

            # Regression target
            y_train_r = tr["future_return"].astype(float)
            y_test_r = te["future_return"].astype(float)

            ridge = Pipeline(
                steps=[
                    (
                        "prep",
                        ColumnTransformer(
                            transformers=[
                                (
                                    "num",
                                    Pipeline(
                                        steps=[
                                            ("imputer", SimpleImputer(strategy="median")),
                                            ("scaler", StandardScaler()),
                                        ]
                                    ),
                                    feature_cols,
                                )
                            ]
                        ),
                    ),
                    ("reg", Ridge(alpha=1.0, random_state=42)),
                ]
            )
            ridge.fit(X_train, y_train_r)
            ridge_pred = ridge.predict(X_test)

            gbr = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "reg",
                        HistGradientBoostingRegressor(
                            max_depth=4,
                            learning_rate=0.05,
                            max_iter=300,
                            random_state=42,
                        ),
                    ),
                ]
            )
            gbr.fit(X_train, y_train_r)
            gbr_pred = gbr.predict(X_test)
            zero_pred = np.zeros_like(y_test_r)

            metrics_rows.extend(
                self._regression_metrics(y_test_r, zero_pred, "kalshi", "movement_return", int(horizon), "baseline_zero")
            )
            metrics_rows.extend(
                self._regression_metrics(y_test_r, ridge_pred, "kalshi", "movement_return", int(horizon), "ridge")
            )
            metrics_rows.extend(
                self._regression_metrics(
                    y_test_r,
                    gbr_pred,
                    "kalshi",
                    "movement_return",
                    int(horizon),
                    "hist_gradient_boosting_regressor",
                )
            )

            for variant, pred in [
                ("baseline_zero", zero_pred),
                ("ridge", ridge_pred),
                ("hgb_regressor", gbr_pred),
            ]:
                ablation_rows.extend(
                    [
                        {
                            "platform": "kalshi",
                            "target": "movement_return",
                            "horizon_hours": int(horizon),
                            "variant": variant,
                            "metric": "mae",
                            "value": float(mean_absolute_error(y_test_r, pred)),
                        },
                        {
                            "platform": "kalshi",
                            "target": "movement_return",
                            "horizon_hours": int(horizon),
                            "variant": variant,
                            "metric": "rmse",
                            "value": float(np.sqrt(mean_squared_error(y_test_r, pred))),
                        },
                    ]
                )

            # Direction classification
            y_train_c = tr["future_up"].astype(int)
            y_test_c = te["future_up"].astype(int)
            if y_train_c.nunique() < 2 or y_test_c.nunique() < 2:
                continue

            baseline_prob = (te["ret_1"] > 0).astype(float).to_numpy() * 0.9 + 0.05

            logit = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
                ]
            )
            logit.fit(X_train, y_train_c)
            logit_p = logit.predict_proba(X_test)[:, 1].clip(1e-4, 1 - 1e-4)

            hgbc = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_depth=4,
                            learning_rate=0.05,
                            max_iter=300,
                            random_state=42,
                        ),
                    ),
                ]
            )
            hgbc.fit(X_train, y_train_c)
            hgbc_p = hgbc.predict_proba(X_test)[:, 1].clip(1e-4, 1 - 1e-4)

            metrics_rows.extend(
                self._classification_metrics(
                    y_test_c,
                    baseline_prob,
                    "kalshi",
                    "movement_direction",
                    int(horizon),
                    "baseline_recent_return_sign",
                )
            )
            metrics_rows.extend(
                self._classification_metrics(y_test_c, logit_p, "kalshi", "movement_direction", int(horizon), "logistic")
            )
            metrics_rows.extend(
                self._classification_metrics(
                    y_test_c,
                    hgbc_p,
                    "kalshi",
                    "movement_direction",
                    int(horizon),
                    "hist_gradient_boosting",
                )
            )

            for variant, prob in [
                ("baseline_recent_return_sign", baseline_prob),
                ("logistic", logit_p),
                ("hgb_classifier", hgbc_p),
            ]:
                ablation_rows.extend(
                    self._ablation_from_classification(
                        y_test_c,
                        prob,
                        "kalshi",
                        "movement_direction",
                        int(horizon),
                        variant,
                    )
                )

        return metrics_rows, ablation_rows

    def _plot_basic_distributions(self):
        con = duckdb.connect()
        trades_glob = str(self.kalshi_trades / "*.parquet")
        summary = con.execute(
            f"""
            WITH per_market AS (
                SELECT ticker,
                       COUNT(*) AS trade_count,
                       SUM(count) AS contracts
                FROM '{trades_glob}'
                GROUP BY ticker
            )
            SELECT
                quantile_cont(yes_price, 0.5) AS price_p50,
                quantile_cont(yes_price, 0.9) AS price_p90,
                quantile_cont(count, 0.5) AS size_p50,
                quantile_cont(count, 0.9) AS size_p90,
                quantile_cont(trade_count, 0.5) AS trades_p50,
                quantile_cont(trade_count, 0.9) AS trades_p90
            FROM '{trades_glob}', per_market
            LIMIT 1
            """
        ).df()

        sampled = con.execute(
            f"""
            SELECT yes_price, count
            FROM '{trades_glob}'
            USING SAMPLE 200000 ROWS
            """
        ).df()
        market_counts = con.execute(
            f"""
            SELECT ticker, COUNT(*) AS trade_count
            FROM '{trades_glob}'
            GROUP BY ticker
            ORDER BY trade_count DESC
            LIMIT 5000
            """
        ).df()

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].hist(sampled["yes_price"], bins=40, color="#4472C4", alpha=0.85)
        axes[0].set_title("Kalshi Yes Price Distribution")
        axes[0].set_xlabel("Yes price (cents)")

        axes[1].hist(sampled["count"].clip(upper=500), bins=40, color="#ED7D31", alpha=0.85)
        axes[1].set_title("Trade Size Distribution")
        axes[1].set_xlabel("Contracts (clipped at 500)")

        axes[2].hist(market_counts["trade_count"], bins=40, color="#70AD47", alpha=0.85)
        axes[2].set_title("Trade Count per Market")
        axes[2].set_xlabel("Trades per market")

        note = ""
        if not summary.empty:
            row = summary.iloc[0]
            note = (
                f"price p50/p90={row['price_p50']:.1f}/{row['price_p90']:.1f} | "
                f"size p50/p90={row['size_p50']:.1f}/{row['size_p90']:.1f} | "
                f"trades p50/p90={row['trades_p50']:.1f}/{row['trades_p90']:.1f}"
            )
        fig.suptitle(f"Distribution Overview. {note}" if note else "Distribution Overview", y=1.02)
        fig.tight_layout()
        fig.savefig(self.output_dir / "prediction_modeling_distributions.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_time_series_examples(self):
        con = duckdb.connect()
        trades_glob = str(self.kalshi_trades / "*.parquet")
        markets_glob = str(self.kalshi_markets / "*.parquet")
        df = con.execute(
            f"""
            WITH latest_markets AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker
                           ORDER BY COALESCE(_fetched_at, close_time, created_time) DESC
                       ) AS rn
                FROM '{markets_glob}'
            ),
            resolved AS (
                SELECT ticker
                FROM latest_markets
                WHERE rn = 1
                  AND result IN ('yes', 'no')
            ),
            top_tickers AS (
                SELECT t.ticker, COUNT(*) AS n
                FROM '{trades_glob}' t
                INNER JOIN resolved r ON t.ticker = r.ticker
                GROUP BY 1
                ORDER BY n DESC
                LIMIT 3
            )
            SELECT t.ticker, t.created_time, t.yes_price
            FROM '{trades_glob}' t
            INNER JOIN top_tickers tt ON t.ticker = tt.ticker
            WHERE t.yes_price BETWEEN 1 AND 99
            ORDER BY t.ticker, t.created_time
            """
        ).df()

        if df.empty:
            return

        df["created_time"] = pd.to_datetime(df["created_time"])

        fig, ax = plt.subplots(figsize=(11, 5))
        for ticker, group in df.groupby("ticker"):
            g = group.sort_values("created_time")
            ax.plot(g["created_time"], g["yes_price"] / 100.0, label=str(ticker), linewidth=1.2)
        ax.set_title("Kalshi Price Time Series Examples (Most Active Markets)")
        ax.set_ylabel("Yes price")
        ax.set_xlabel("Time")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.output_dir / "prediction_modeling_time_series_examples.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_calibration_curve(
        self,
        y_true: pd.Series,
        baseline_p: np.ndarray,
        logistic_p: np.ndarray,
        gbm_p: np.ndarray,
        cal_p: np.ndarray,
    ):
        fig, ax = plt.subplots(figsize=(7, 6))
        for name, probs in [
            ("Baseline (last price)", baseline_p),
            ("Logistic", logistic_p),
            ("HGB", gbm_p),
            ("HGB + Isotonic", cal_p),
        ]:
            frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", label=name)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title("Outcome Prediction Calibration")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.output_dir / "prediction_modeling_outcome_calibration.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_roc_curve(
        self,
        y_true: pd.Series,
        baseline_p: np.ndarray,
        logistic_p: np.ndarray,
        gbm_p: np.ndarray,
        cal_p: np.ndarray,
    ):
        fig, ax = plt.subplots(figsize=(7, 6))
        for name, probs in [
            ("Baseline (last price)", baseline_p),
            ("Logistic", logistic_p),
            ("HGB", gbm_p),
            ("HGB + Isotonic", cal_p),
        ]:
            fpr, tpr, _ = roc_curve(y_true, probs)
            auc = roc_auc_score(y_true, probs)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_title("Outcome Prediction ROC")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.output_dir / "prediction_modeling_outcome_roc.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_outcome_feature_importance(
        self,
        gbm_pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_cols: list[str],
    ):
        perm = permutation_importance(
            gbm_pipeline,
            X_test,
            y_test,
            n_repeats=8,
            random_state=42,
            scoring="roc_auc",
        )
        order = np.argsort(perm.importances_mean)[::-1]

        imp_df = pd.DataFrame(
            {
                "feature": np.array(feature_cols)[order],
                "importance_mean": perm.importances_mean[order],
                "importance_std": perm.importances_std[order],
            }
        )
        imp_df.to_csv(self.output_dir / "prediction_modeling_outcome_feature_importance.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            imp_df["feature"].iloc[::-1],
            imp_df["importance_mean"].iloc[::-1],
            xerr=imp_df["importance_std"].iloc[::-1],
            color="#5B9BD5",
            alpha=0.9,
        )
        ax.set_title("Outcome Model Feature Importance (Permutation)")
        ax.set_xlabel("Mean decrease in AUC")
        fig.tight_layout()
        fig.savefig(self.output_dir / "prediction_modeling_outcome_feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _classification_metrics(
        self,
        y_true: pd.Series,
        y_prob: np.ndarray,
        platform: str,
        target: str,
        horizon_hours: int | str,
        model: str,
    ) -> list[dict[str, object]]:
        y_pred = (y_prob >= 0.5).astype(int)
        rows = [
            {
                "platform": platform,
                "target": target,
                "horizon_hours": horizon_hours,
                "model": model,
                "metric": "brier",
                "value": float(brier_score_loss(y_true, y_prob)),
            },
            {
                "platform": platform,
                "target": target,
                "horizon_hours": horizon_hours,
                "model": model,
                "metric": "log_loss",
                "value": float(log_loss(y_true, y_prob)),
            },
            {
                "platform": platform,
                "target": target,
                "horizon_hours": horizon_hours,
                "model": model,
                "metric": "accuracy",
                "value": float(accuracy_score(y_true, y_pred)),
            },
        ]
        if len(np.unique(y_true)) > 1:
            rows.append(
                {
                    "platform": platform,
                    "target": target,
                    "horizon_hours": horizon_hours,
                    "model": model,
                    "metric": "auc",
                    "value": float(roc_auc_score(y_true, y_prob)),
                }
            )
        return rows

    def _ablation_from_classification(
        self,
        y_true: pd.Series,
        y_prob: np.ndarray,
        platform: str,
        target: str,
        horizon_hours: int | str,
        variant: str,
    ) -> list[dict[str, object]]:
        return [
            {
                "platform": platform,
                "target": target,
                "horizon_hours": horizon_hours,
                "variant": variant,
                "metric": "brier",
                "value": float(brier_score_loss(y_true, y_prob)),
            },
            {
                "platform": platform,
                "target": target,
                "horizon_hours": horizon_hours,
                "variant": variant,
                "metric": "log_loss",
                "value": float(log_loss(y_true, y_prob)),
            },
            {
                "platform": platform,
                "target": target,
                "horizon_hours": horizon_hours,
                "variant": variant,
                "metric": "auc",
                "value": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
            },
        ]

    def _regression_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        platform: str,
        target: str,
        horizon_hours: int,
        model: str,
    ) -> list[dict[str, object]]:
        return [
            {
                "platform": platform,
                "target": target,
                "horizon_hours": horizon_hours,
                "model": model,
                "metric": "mae",
                "value": float(mean_absolute_error(y_true, y_pred)),
            },
            {
                "platform": platform,
                "target": target,
                "horizon_hours": horizon_hours,
                "model": model,
                "metric": "rmse",
                "value": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            },
        ]

    def _split_market_time(
        self,
        df: pd.DataFrame,
        market_col: str,
        time_col: str,
        train_frac: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        market_times = df.groupby(market_col)[time_col].max().sort_values()
        if market_times.empty:
            return df.iloc[0:0].copy(), df.iloc[0:0].copy()

        cutoff_idx = max(1, int(len(market_times) * train_frac))
        train_markets = set(market_times.iloc[:cutoff_idx].index)

        train_df = df[df[market_col].isin(train_markets)].copy()
        test_df = df[~df[market_col].isin(train_markets)].copy()
        return train_df, test_df

    def _write_summary_md(self, kalshi_profile: dict[str, object], polymarket_profile: dict[str, object], metrics_df: pd.DataFrame):
        summary_path = self.output_dir / "summary.md"

        if metrics_df.empty:
            metrics_text = "No model metrics produced. Data was missing or insufficient for leakage-safe train/test splits."
        else:
            top = (
                metrics_df.sort_values(["target", "horizon_hours", "metric", "value"])
                .groupby(["target", "horizon_hours", "metric"], as_index=False)
                .first()
            )
            metrics_text = top.to_markdown(index=False)

        content = f"""# Prediction Modeling Pipeline Summary

## What Was Run
- Data profiling for Kalshi and Polymarket parquet datasets.
- Target A (classification): Kalshi market `result == yes` prediction from pre-resolution snapshots at {self.outcome_horizons} hours before close.
- Target B (regression + classification): Kalshi short-horizon price movement over {self.movement_horizons} hours.

## Leakage Prevention
- Outcome target uses only trades with `created_time <= snapshot_time` and snapshots strictly before market `close_time`.
- Time split for outcome prediction is market-level: markets sorted by close time, then earlier markets for training and later markets for testing.
- Movement target uses chronological split by trade timestamp (train early period, test later period).
- No post-resolution columns are used in feature construction.

## Data Used
- Kalshi datasets available: {', '.join(kalshi_profile.get('datasets_available', [])) or 'none'}
- Polymarket datasets available: {', '.join(polymarket_profile.get('datasets_available', [])) or 'none'}

## Key Metrics (Best-by-group table)
{metrics_text}

## Caveats
- This is a first-pass pipeline optimized for reproducibility and leakage controls.
- Kalshi trader concentration / HHI features were omitted because trader identifiers are not present in the Kalshi trade schema.
- Polymarket modeling is not included in this pass due complex token-to-market resolution dependencies; profiling outputs are included for both platforms.

## Next Steps
- Add Polymarket token-resolution joins for outcome labels, then run the same target A/B framework per platform.
- Add walk-forward validation windows for stronger robustness.
- Expand feature engineering with spread proxies from quote snapshots where available.
"""
        summary_path.write_text(content)

    def _write_missing_data_summary(self):
        (self.output_dir / "summary.md").write_text(
            "# Prediction Modeling Pipeline Summary\n\n"
            "Data directory is missing or incomplete.\n\n"
            "Expected parquet directories:\n"
            "- data/kalshi/markets\n"
            "- data/kalshi/trades\n"
            "- data/polymarket/markets\n"
            "- data/polymarket/trades\n\n"
            "Run `make setup` to download the dataset, then rerun `make analyze` and select `prediction_modeling_pipeline`.\n"
        )

    def _field_semantics(self, platform: str) -> dict[str, str]:
        if platform == "kalshi":
            return {
                "price": "yes_price/no_price are in cents (1-99), implying probabilities in [0.01, 0.99].",
                "side": "taker_side indicates whether taker bought YES or NO.",
                "quantity": "count is the number of contracts traded.",
                "timestamps": "created_time marks trade execution; close_time marks market close/resolution horizon.",
                "resolution": "result in {yes,no} for finalized markets.",
            }
        return {
            "price": "trade price is derived from maker/taker USDC vs outcome token amounts (0-1 scale).",
            "side": "trade direction depends on whether USDC is maker or taker asset.",
            "quantity": "maker_amount/taker_amount encode token and collateral quantities.",
            "timestamps": "trades use block_number; timestamps can be joined via blocks dataset.",
            "resolution": "resolved outcome can be inferred from terminal outcome_prices snapshots in markets data.",
        }

    @staticmethod
    def _qid(name: str) -> str:
        return f'"{name}"'

    @staticmethod
    def _pick_first(candidates: list[str], preference: list[str]) -> str | None:
        for col in preference:
            if col in candidates:
                return col
        return None

    @staticmethod
    def _has_parquet(path: Path) -> bool:
        return path.exists() and any(path.glob("*.parquet"))
