"""
IE7374 - Hyperparameter Optimization Lab
Fashion MNIST Classification:
  Baseline → Keras Tuner (RandomSearch, Hyperband, BayesianOptimization) → Optuna (TPE)

Usage:
  python hpo_fashion_mnist.py --mode baseline
  python hpo_fashion_mnist.py --mode keras-tuner --kt-strategy hyperband
  python hpo_fashion_mnist.py --mode keras-tuner --kt-strategy random
  python hpo_fashion_mnist.py --mode keras-tuner --kt-strategy bayesian
  python hpo_fashion_mnist.py --mode optuna
  python hpo_fashion_mnist.py --mode all
"""

import argparse
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import optuna
from optuna_integration.tfkeras import TFKerasPruningCallback

# Configuration

def parse_args():
    parser = argparse.ArgumentParser(
        description="HPO Lab: Baseline vs Keras Tuner vs Optuna on Fashion MNIST"
    )

    # Mode
    parser.add_argument("--mode", type=str, default="all",
                        choices=["baseline", "keras-tuner", "optuna", "all"],
                        help="Which pipeline to run (default: all)")

    # Training
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--validation-split", type=float, default=0.2,
                        help="Validation split (default: 0.2)")
    parser.add_argument("--early-stop-patience", type=int, default=5,
                        help="Early stopping patience (default: 5)")

    # Keras Tuner settings
    parser.add_argument("--kt-strategy", type=str, nargs="+",
                        default=["hyperband", "random", "bayesian"],
                        choices=["hyperband", "random", "bayesian"],
                        help="Keras Tuner strategies to run (default: all three)")
    parser.add_argument("--kt-factor", type=int, default=3,
                        help="Hyperband reduction factor (default: 3)")
    parser.add_argument("--kt-max-epochs", type=int, default=10,
                        help="Hyperband max epochs per trial (default: 10)")
    parser.add_argument("--kt-max-trials", type=int, default=20,
                        help="Max trials for RandomSearch/Bayesian (default: 20)")
    parser.add_argument("--kt-overwrite", action="store_true",
                        help="Overwrite previous Keras Tuner results")

    # Optuna settings
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Number of Optuna trials (default: 20)")

    # Search space (shared by all tuners)
    parser.add_argument("--units-min", type=int, default=32,
                        help="Min Dense units (default: 32)")
    parser.add_argument("--units-max", type=int, default=512,
                        help="Max Dense units (default: 512)")
    parser.add_argument("--units-step", type=int, default=32,
                        help="Step for Dense units (default: 32)")
    parser.add_argument("--lr-choices", type=float, nargs="+",
                        default=[1e-2, 1e-3, 1e-4],
                        help="Learning rate choices (default: 0.01 0.001 0.0001)")
    parser.add_argument("--dropout-min", type=float, default=0.1,
                        help="Min dropout (Optuna only, default: 0.1)")
    parser.add_argument("--dropout-max", type=float, default=0.5,
                        help="Max dropout (Optuna only, default: 0.5)")

    # Output
    parser.add_argument("--output-file", type=str, default="results.txt",
                        help="File to append results to (default: results.txt)")
    parser.add_argument("--log-dir", type=str, default="./tb_logs",
                        help="TensorBoard log directory (default: ./tb_logs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    return parser.parse_args()


# Logging

def log(msg, file=None):
    print(msg)
    if file:
        file.write(msg + "\n")


def print_results(model_name, params: dict, eval_dict: dict, file=None):
    log(f"\n{'='*50}", file)
    log(f" {model_name}", file)
    log(f"{'='*50}", file)
    for k, v in params.items():
        log(f"  {k}: {v}", file)
    log(f"  {'─'*40}", file)
    for k, v in eval_dict.items():
        log(f"  {k}: {v:.4f}", file)
    log("", file)


# Data

def load_data():
    (img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()
    img_train = img_train.astype("float32") / 255.0
    img_test = img_test.astype("float32") / 255.0
    return (img_train, label_train), (img_test, label_test)


# Shared Keras Tuner model builder

def create_kt_model_builder(args):
    def model_builder(hp):
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(28, 28)))
        hp_units = hp.Int("units", min_value=args.units_min,
                          max_value=args.units_max, step=args.units_step)
        model.add(keras.layers.Dense(units=hp_units, activation="relu",
                                     name="tuned_dense_1"))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(10, activation="softmax"))
        hp_lr = hp.Choice("learning_rate", values=args.lr_choices)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_lr),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        return model
    return model_builder


# 1. Baseline

def run_baseline(args, train_data, test_data, f):
    img_train, label_train = train_data
    img_test, label_test = test_data

    log("\n>>> [BASELINE] Training with hardcoded hyperparameters...", f)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(512, activation="relu", name="dense_1"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(
        img_train, label_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        callbacks=[keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.log_dir, "baseline"), update_freq="epoch"
        )],
        verbose=1,
    )

    eval_dict = model.evaluate(img_test, label_test, return_dict=True, verbose=0)
    print_results(
        "BASELINE MODEL",
        {"units": 512, "learning_rate": 0.001, "dropout": 0.2, "activation": "relu"},
        eval_dict, f,
    )
    return eval_dict


# 2. Keras Tuner (multiple strategies)

def run_keras_tuner(args, train_data, test_data, f):
    img_train, label_train = train_data
    img_test, label_test = test_data

    model_builder = create_kt_model_builder(args)
    strategies = args.kt_strategy

    for strategy in strategies:
        log(f"\n>>> [KERAS TUNER - {strategy.upper()}] Starting search...", f)
        log(f"    Search space:", f)
        log(f"      units : [{args.units_min}, {args.units_max}] step {args.units_step}", f)
        log(f"      lr    : {args.lr_choices}", f)

        # Create the tuner based on strategy
        if strategy == "hyperband":
            tuner = kt.Hyperband(
                model_builder,
                objective="val_accuracy",
                max_epochs=args.kt_max_epochs,
                factor=args.kt_factor,
                directory="kt_dir",
                project_name="kt_hyperband",
                overwrite=args.kt_overwrite,
            )
        elif strategy == "random":
            tuner = kt.RandomSearch(
                model_builder,
                objective="val_accuracy",
                max_trials=args.kt_max_trials,
                directory="kt_dir",
                project_name="kt_random",
                overwrite=args.kt_overwrite,
            )
        elif strategy == "bayesian":
            tuner = kt.BayesianOptimization(
                model_builder,
                objective="val_accuracy",
                max_trials=args.kt_max_trials,
                directory="kt_dir",
                project_name="kt_bayesian",
                overwrite=args.kt_overwrite,
            )

        tuner.search_space_summary()

        tuner.search(
            img_train, label_train,
            epochs=args.epochs,
            validation_split=args.validation_split,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=args.early_stop_patience
                ),
                keras.callbacks.TensorBoard(
                    log_dir=os.path.join(args.log_dir, f"kt_{strategy}_search"),
                    update_freq="batch",
                ),
            ],
        )

        best_hps = tuner.get_best_hyperparameters()[0]
        best_units = best_hps.get("units")
        best_lr = best_hps.get("learning_rate")
        log(f"    Best units: {best_units}", f)
        log(f"    Best learning rate: {best_lr}", f)

        # Retrain best model
        log(f"\n>>> [KERAS TUNER - {strategy.upper()}] Retraining best model...", f)
        h_model = tuner.hypermodel.build(best_hps)
        h_model.fit(
            img_train, label_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            callbacks=[keras.callbacks.TensorBoard(
                log_dir=os.path.join(args.log_dir, f"kt_{strategy}_best"),
                update_freq="epoch",
            )],
            verbose=1,
        )

        eval_dict = h_model.evaluate(img_test, label_test, return_dict=True, verbose=0)
        print_results(
            f"KERAS TUNER - {strategy.upper()}",
            {"units": best_units, "learning_rate": best_lr, "dropout": 0.2},
            eval_dict, f,
        )


# 3. Optuna (TPE + MedianPruner)

def run_optuna(args, train_data, test_data, f):
    img_train, label_train = train_data
    img_test, label_test = test_data

    log(f"\n>>> [OPTUNA] Starting TPE search ({args.n_trials} trials)...", f)
    log(f"    Search space:", f)
    log(f"      units      : [{args.units_min}, {args.units_max}] step {args.units_step}", f)
    log(f"      lr         : {args.lr_choices}", f)
    log(f"      dropout    : [{args.dropout_min}, {args.dropout_max}]", f)
    log(f"      activation : [relu, tanh, selu]", f)

    def objective(trial: optuna.Trial) -> float:
        units = trial.suggest_int("units", args.units_min, args.units_max, step=args.units_step)
        lr = trial.suggest_categorical("learning_rate", args.lr_choices)
        dropout = trial.suggest_float("dropout_rate", args.dropout_min, args.dropout_max, step=0.05)
        activation = trial.suggest_categorical("activation", ["relu", "tanh", "selu"])

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(units, activation=activation, name="tuned_dense_1"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(10, activation="softmax"),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        history = model.fit(
            img_train, label_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=args.early_stop_patience,
                    restore_best_weights=True
                ),
                TFKerasPruningCallback(trial, monitor="val_accuracy"),
                keras.callbacks.TensorBoard(
                    log_dir=os.path.join(args.log_dir, f"optuna_trial_{trial.number}"),
                    update_freq="epoch",
                ),
            ],
            verbose=0,
        )
        return max(history.history["val_accuracy"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        study_name="fashion_mnist_optuna",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Study summary
    log(f"\n    Best trial    : {study.best_trial.number}", f)
    log(f"    Best val_acc  : {study.best_value:.4f}", f)
    for k, v in study.best_params.items():
        log(f"    Best {k}: {v}", f)
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    log(f"    Completed: {complete} | Pruned: {pruned}", f)

    # Retrain best model
    log("\n>>> [OPTUNA] Retraining best model...", f)
    best = study.best_params
    best_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(best["units"], activation=best["activation"], name="tuned_dense_1"),
        keras.layers.Dropout(best["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])
    best_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best["learning_rate"]),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    best_model.fit(
        img_train, label_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.early_stop_patience,
                restore_best_weights=True
            ),
            keras.callbacks.TensorBoard(
                log_dir=os.path.join(args.log_dir, "optuna_best"), update_freq="epoch"
            ),
        ],
        verbose=1,
    )

    eval_dict = best_model.evaluate(img_test, label_test, return_dict=True, verbose=0)
    print_results("OPTUNA TPE MODEL", best, eval_dict, f)


# Main

def main():
    args = parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    f = open(args.output_file, "a")
    header = (f"\n{'#'*60}\n"
              f"# Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
              f"# Mode: {args.mode}\n"
              f"# Args: {vars(args)}\n"
              f"{'#'*60}")
    log(header, f)

    log("Loading Fashion MNIST dataset...", f)
    train_data, test_data = load_data()

    if args.mode in ("baseline", "all"):
        run_baseline(args, train_data, test_data, f)

    if args.mode in ("keras-tuner", "all"):
        run_keras_tuner(args, train_data, test_data, f)

    if args.mode in ("optuna", "all"):
        run_optuna(args, train_data, test_data, f)

    f.close()
    log(f"\n Results appended to {args.output_file}", None)


if __name__ == "__main__":
    main()