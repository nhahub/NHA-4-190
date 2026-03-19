"""
main.py
-------
Entry point for the Predictive Maintenance pipeline.
Runs Milestones 1–3 end-to-end and saves all artifacts.

Usage:
    python main.py
    python main.py --config config/config.yaml
"""

import argparse
from src.utils import get_logger, load_config
from src.data_preprocessing import run_preprocessing
from src.feature_engineering import run_feature_engineering
from src.model_training import run_training_pipeline
from src.model_evaluation import evaluate_all_models

logger = get_logger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Powered Predictive Maintenance Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip GridSearch (faster run for debugging)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    logger.info("=" * 60)
    logger.info("  PREDICTIVE MAINTENANCE PIPELINE — START")
    logger.info("=" * 60)

    # ── Milestone 1: Preprocessing ──────────────────────────────
    logger.info("\n[M1] Data Collection, Exploration & Preprocessing")
    df_clean = run_preprocessing(args.config)

    # ── Milestone 2: Feature Engineering ────────────────────────
    logger.info("\n[M2] Advanced Feature Engineering")
    df_engineered = run_feature_engineering(df_clean, config)

    # ── Milestone 3: Model Training & Optimization ───────────────
    logger.info("\n[M3] Model Development & Optimization")
    training_results = run_training_pipeline(
        df_engineered,
        config,
        save_models=True,
    )

    # ── Milestone 3: Evaluation ───────────────────────────────────
    logger.info("\n[M3] Model Evaluation")
    models_to_evaluate = {
        "Random Forest (Baseline)":  training_results["rf"],
        "Random Forest (Optimized)": training_results["rf_best"],  # ← the tuned one
        "XGBoost (Baseline)":        training_results["xgb_base"],
    }
    evaluate_all_models(
        models_to_evaluate,
        training_results["X_test"],
        training_results["y_test"],
    )

    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE. Models saved to /models")
    logger.info("  Plots saved to /outputs/plots")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
