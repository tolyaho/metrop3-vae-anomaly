from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import json
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "0")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..evaluation.metrics import binary_metrics, roc_auc_binary

tf.get_logger().setLevel("ERROR")


def seed_everything(seed: int, *, enable_op_determinism: bool | None = None) -> None:
    """Seed Python, NumPy, and TensorFlow for deterministic training.

    By default ``enable_op_determinism`` follows the ``VAE_DETERMINISM``
    environment variable (``"1"``/``"true"`` to enable, anything else to
    disable). The flag is **off by default** because forcing deterministic
    cuDNN kernels can interact poorly with certain weight initialisations
    on this dataset (the VAE collapses into an inverted-score solution),
    so we keep the option but do not impose it.
    """
    if enable_op_determinism is None:
        env_val = os.environ.get("VAE_DETERMINISM", "0").strip().lower()
        enable_op_determinism = env_val in {"1", "true", "yes"}

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
    if enable_op_determinism:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def _serialize_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


@dataclass
class VAEConfig:
    latent_dim: int = 16
    architecture: str = "conv1d"  # conv1d | dense | lstm_autoencoder
    conv_filters: tuple[int, ...] = (32, 64)
    kernel_size: int = 3
    hidden_units: tuple[int, ...] = (64, 32)
    lstm_units: tuple[int, ...] = (64, 32)
    encoder_use_batchnorm: bool = False
    encoder_dropout_rate: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 512
    learning_rate: float = 1e-3
    beta: float = 0
    beta_warmup_epochs: int = 0  # ramp beta from 0->beta over this many epochs; 0 = no warmup
    validation_split: float = 0
    gradient_clipnorm: float = 1.0
    random_seed: int = 42
    verbose_epoch: bool = True
    early_stopping_metric: str = "val_roc_auc"  # val_roc_auc | val_total_loss | none
    early_stopping_patience: int = 5
    early_stopping_min_epochs: int = 2
    early_stopping_restore_best: bool = True


@dataclass
class ThresholdConfig:
    method: str = "val_f1"  # val_f1 | percentile | mean_std
    percentile: float = 95.0
    std_factor: float = 3.0


@tf.keras.utils.register_keras_serializable(package="vae_anomaly_detector")
class Sampling(layers.Layer):
    def call(self, inputs, training=False):
        mu, log_var = inputs
        if not training:
            return mu
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps


def build_vae(window_size: int, n_features: int, cfg: VAEConfig):
    encoder_inputs = keras.Input(shape=(window_size, n_features), name="encoder_input")
    x = encoder_inputs

    def apply_encoder_hidden_block(inputs, units: int, layer_index: int, total_hidden_layers: int):
        y = layers.Dense(units, activation=None, name=f"enc_dense_{layer_index+1}")(inputs)
        if cfg.encoder_use_batchnorm:
            y = layers.BatchNormalization(name=f"enc_bn_{layer_index+1}")(y)
        y = layers.ReLU(name=f"enc_relu_{layer_index+1}")(y)
        if layer_index < total_hidden_layers - 1 and cfg.encoder_dropout_rate > 0:
            y = layers.Dropout(cfg.encoder_dropout_rate, name=f"enc_dropout_{layer_index+1}")(y)
        return y

    if cfg.architecture == "conv1d":
        for i, filters in enumerate(cfg.conv_filters):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=cfg.kernel_size,
                padding="same",
                activation="relu",
                name=f"enc_conv1d_{i+1}",
            )(x)
        x = layers.Flatten(name="flatten_input")(x)
        for i, units in enumerate(cfg.hidden_units):
            x = apply_encoder_hidden_block(x, units, i, len(cfg.hidden_units))
    elif cfg.architecture == "lstm_autoencoder":
        # LSTM encoder: keep temporal structure and compress to a fixed vector.
        for i, units in enumerate(cfg.lstm_units[:-1]):
            x = layers.LSTM(units, return_sequences=True, name=f"enc_lstm_{i+1}")(x)
        x = layers.LSTM(cfg.lstm_units[-1], return_sequences=False, name=f"enc_lstm_{len(cfg.lstm_units)}")(x)
        for i, units in enumerate(cfg.hidden_units):
            x = apply_encoder_hidden_block(x, units, i, len(cfg.hidden_units))
    else:
        x = layers.Flatten(name="flatten_input")(x)
        for i, units in enumerate(cfg.hidden_units):
            x = apply_encoder_hidden_block(x, units, i, len(cfg.hidden_units))

    mu = layers.Dense(cfg.latent_dim, name="z_mean")(x)
    log_var = layers.Dense(cfg.latent_dim, name="z_log_var")(x)
    z = Sampling(name="z")([mu, log_var])
    encoder = keras.Model(encoder_inputs, [mu, log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(cfg.latent_dim,), name="decoder_input")
    y = latent_inputs
    for i, units in enumerate(reversed(cfg.hidden_units)):
        y = layers.Dense(units, activation="relu", name=f"dec_dense_{i+1}")(y)
    if cfg.architecture == "conv1d":
        y = layers.Dense(window_size * cfg.conv_filters[-1], activation="relu", name="dec_dense_to_seq")(y)
        y = layers.Reshape((window_size, cfg.conv_filters[-1]), name="dec_reshape_seq")(y)
        for i, filters in enumerate(reversed(cfg.conv_filters[:-1])):
            y = layers.Conv1D(
                filters=filters,
                kernel_size=cfg.kernel_size,
                padding="same",
                activation="relu",
                name=f"dec_conv1d_{i+1}",
            )(y)
        decoder_outputs = layers.Conv1D(
            filters=n_features,
            kernel_size=cfg.kernel_size,
            padding="same",
            activation="linear",
            name="decoder_output",
        )(y)
    elif cfg.architecture == "lstm_autoencoder":
        # LSTM decoder: expand latent to sequence, then reconstruct each time step.
        y = layers.RepeatVector(window_size, name="dec_repeat_vector")(y)
        for i, units in enumerate(reversed(cfg.lstm_units)):
            y = layers.LSTM(units, return_sequences=True, name=f"dec_lstm_{i+1}")(y)
        decoder_outputs = layers.TimeDistributed(
            layers.Dense(n_features, activation="linear"), name="decoder_output"
        )(y)
    else:
        y = layers.Dense(window_size * n_features, activation="linear", name="decoder_output_flat")(y)
        decoder_outputs = layers.Reshape((window_size, n_features), name="decoder_output")(y)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return encoder, decoder


def _split_train_val(x: np.ndarray, val_ratio: float, seed: int):
    if val_ratio <= 0.0:
        return x, np.empty((0,) + x.shape[1:], dtype=x.dtype)
    n = len(x)
    n_val = max(1, int(n * val_ratio))
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return x[train_idx], x[val_idx]


def train_vae(
    train_windows: np.ndarray,
    vae_cfg: VAEConfig,
    train_cfg: TrainConfig,
    val_windows: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
):
    if len(train_windows) == 0:
        raise ValueError(
            "No training windows provided. Run scripts/build_windows.py first."
        )

    seed_everything(train_cfg.random_seed)

    window_size = train_windows.shape[1]
    n_features = train_windows.shape[2]
    encoder, decoder = build_vae(window_size, n_features, vae_cfg)

    optimizer_kwargs = {"learning_rate": train_cfg.learning_rate}
    if train_cfg.gradient_clipnorm is not None and train_cfg.gradient_clipnorm > 0:
        optimizer_kwargs["clipnorm"] = float(train_cfg.gradient_clipnorm)
    optimizer = keras.optimizers.Adam(**optimizer_kwargs)

    if val_windows is not None and len(val_windows):
        x_train = train_windows
        x_val = val_windows
    else:
        x_train, x_val = _split_train_val(train_windows, train_cfg.validation_split, train_cfg.random_seed)

        if len(x_train) == 0:
            # If validation split consumed all data (tiny datasets), fall back to full train.
            x_train = train_windows
            x_val = np.empty((0,) + train_windows.shape[1:], dtype=train_windows.dtype)

    train_ds = tf.data.Dataset.from_tensor_slices(x_train)
    if len(x_train) > 1:
        train_ds = train_ds.shuffle(len(x_train), seed=train_cfg.random_seed)
    train_ds = train_ds.batch(train_cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices(x_val).batch(train_cfg.batch_size) if len(x_val) else None

    history = {
        "epoch": [],
        "train_total_loss": [],
        "train_recon_loss": [],
        "train_kl_loss": [],
        "val_total_loss": [],
        "val_recon_loss": [],
        "val_kl_loss": [],
        "val_roc_auc": [],
    }

    early_stop_metric = (train_cfg.early_stopping_metric or "none").lower()
    if early_stop_metric not in {"val_roc_auc", "val_total_loss", "none"}:
        raise ValueError(
            f"Unknown early_stopping_metric={early_stop_metric!r}; expected val_roc_auc | val_total_loss | none."
        )
    can_track_auc = (
        val_labels is not None
        and val_windows is not None
        and len(val_windows) == len(val_labels)
        and np.unique(np.asarray(val_labels)).size >= 2
    )
    if early_stop_metric == "val_roc_auc" and not can_track_auc:
        early_stop_metric = "val_total_loss" if val_ds is not None else "none"

    best_metric: float | None = None
    best_epoch = 0
    epochs_since_best = 0
    best_auc: float = -1.0
    best_auc_epoch = 0
    best_auc_encoder_weights: list[np.ndarray] | None = None
    best_auc_decoder_weights: list[np.ndarray] | None = None

    early_stop_info: dict[str, float | int | str | bool] = {
        "metric": early_stop_metric,
        "patience": int(train_cfg.early_stopping_patience),
        "min_epochs": int(train_cfg.early_stopping_min_epochs),
        "stopped_early": False,
        "stopped_reason": "",
        "best_epoch": 0,
        "best_value": float("nan"),
        "best_auc_epoch": 0,
        "best_auc": float("nan"),
        "restored_from": "",
        "completed_epochs": 0,
    }

    current_beta = tf.Variable(train_cfg.beta, dtype=tf.float32, trainable=False)

    def train_step(batch_x):
        with tf.GradientTape() as tape:
            mu, log_var, z = encoder(batch_x, training=True)
            x_hat = decoder(z, training=True)

            recon = tf.reduce_mean(tf.math.squared_difference(batch_x, x_hat), axis=[1, 2])
            kl = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
            total = tf.reduce_mean(recon + current_beta * kl)

        vars_ = encoder.trainable_weights + decoder.trainable_weights
        grads = tape.gradient(total, vars_)
        optimizer.apply_gradients(zip(grads, vars_))
        return tf.reduce_mean(recon), tf.reduce_mean(kl), total

    def eval_step(batch_x):
        mu, log_var, z = encoder(batch_x, training=False)
        x_hat = decoder(z, training=False)
        recon = tf.reduce_mean(tf.math.squared_difference(batch_x, x_hat), axis=[1, 2])
        kl = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
        total = tf.reduce_mean(recon + current_beta * kl)
        return tf.reduce_mean(recon), tf.reduce_mean(kl), total

    for epoch in range(1, train_cfg.epochs + 1):
        # Beta warmup: linearly ramp from 0 to target beta over beta_warmup_epochs.
        warmup = train_cfg.beta_warmup_epochs
        if warmup > 0 and epoch <= warmup:
            beta_val = float(train_cfg.beta) * (epoch / warmup)
        else:
            beta_val = float(train_cfg.beta)
        current_beta.assign(beta_val)

        tr_recon, tr_kl, tr_total, tr_n = 0.0, 0.0, 0.0, 0
        for batch in train_ds:
            r, k, t = train_step(batch)
            tr_recon += float(r)
            tr_kl += float(k)
            tr_total += float(t)
            tr_n += 1

        if val_ds is not None:
            va_recon, va_kl, va_total, va_n = 0.0, 0.0, 0.0, 0
            for batch in val_ds:
                r, k, t = eval_step(batch)
                va_recon += float(r)
                va_kl += float(k)
                va_total += float(t)
                va_n += 1
            va_recon /= max(1, va_n)
            va_kl /= max(1, va_n)
            va_total /= max(1, va_n)
        else:
            va_recon, va_kl, va_total = np.nan, np.nan, np.nan

        tr_recon /= max(1, tr_n)
        tr_kl /= max(1, tr_n)
        tr_total /= max(1, tr_n)

        if can_track_auc and val_ds is not None:
            val_scores_ep = mse_reconstruction_scores(
                encoder, decoder, val_windows, batch_size=train_cfg.batch_size
            )
            try:
                va_auc = roc_auc_binary(np.asarray(val_labels, dtype=np.int32), val_scores_ep)
            except Exception:
                va_auc = float("nan")
        else:
            va_auc = float("nan")

        history["epoch"].append(epoch)
        history["train_total_loss"].append(tr_total)
        history["train_recon_loss"].append(tr_recon)
        history["train_kl_loss"].append(tr_kl)
        history["val_total_loss"].append(va_total)
        history["val_recon_loss"].append(va_recon)
        history["val_kl_loss"].append(va_kl)
        history["val_roc_auc"].append(va_auc)

        # Track the best validation AUC across training. Use ``>=`` so we
        # prefer the LATER epoch on ties / numerical plateaus -- in practice
        # this gives a more converged model with the same separation power.
        if can_track_auc and np.isfinite(va_auc) and va_auc >= best_auc - 1e-6:
            if va_auc > best_auc - 1e-6:
                best_auc = max(float(va_auc), best_auc)
                best_auc_epoch = epoch
                if train_cfg.early_stopping_restore_best:
                    best_auc_encoder_weights = [w.copy() for w in encoder.get_weights()]
                    best_auc_decoder_weights = [w.copy() for w in decoder.get_weights()]

        candidate: float | None = None
        if early_stop_metric == "val_roc_auc" and np.isfinite(va_auc):
            candidate = float(va_auc)
            improved = best_metric is None or candidate > best_metric
        elif early_stop_metric == "val_total_loss" and np.isfinite(va_total):
            candidate = -float(va_total)
            improved = best_metric is None or candidate > best_metric
        else:
            improved = False

        if improved and candidate is not None:
            best_metric = candidate
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        # Catastrophic-collapse guard: stop instantly if val AUC drops below
        # 0.5 after we already saw a high-AUC epoch, restoring the best AUC
        # weights. This protects against the VAE's known anomaly-easier-to-
        # reconstruct attractor on this dataset.
        if can_track_auc and np.isfinite(va_auc) and best_auc > 0.7 and va_auc < 0.5:
            early_stop_info["stopped_early"] = True
            early_stop_info["stopped_reason"] = (
                f"val_roc_auc collapsed to {va_auc:.3f} after peaking at {best_auc:.3f} (epoch {best_auc_epoch})."
            )
            if train_cfg.verbose_epoch:
                print(early_stop_info["stopped_reason"] + " Stopping training.")
            break

        if train_cfg.verbose_epoch:
            base = (
                f"Epoch {epoch}/{train_cfg.epochs} - "
                f"train_total={tr_total:.6f}, train_recon={tr_recon:.6f}, train_kl={tr_kl:.6f}"
            )
            if np.isfinite(va_total):
                base += f", val_total={va_total:.6f}, val_recon={va_recon:.6f}, val_kl={va_kl:.6f}"
            if np.isfinite(va_auc):
                base += f", val_roc_auc={va_auc:.4f}"
            print(base)

        if not np.isfinite(tr_total):
            raise ValueError(
                f"Training loss became non-finite at epoch {epoch}. "
                "Try lowering learning_rate, reducing model size, or increasing gradient clipping."
            )
        if val_ds is not None and not np.isfinite(va_total):
            raise ValueError(
                f"Validation loss became non-finite at epoch {epoch}. "
                "Check validation windows for invalid values and tune learning settings."
            )

        if (
            early_stop_metric != "none"
            and epoch >= max(int(train_cfg.early_stopping_min_epochs), 1)
            and epochs_since_best >= int(train_cfg.early_stopping_patience)
        ):
            early_stop_info["stopped_early"] = True
            early_stop_info["stopped_reason"] = (
                f"no improvement in {early_stop_metric} for {epochs_since_best} epochs (best epoch {best_epoch})."
            )
            if train_cfg.verbose_epoch:
                print(f"Early stopping at epoch {epoch}: " + early_stop_info["stopped_reason"])
            break

    # Always restore the best-AUC checkpoint when one was tracked. This
    # protects the deployed model from the late-stage inversion that this
    # particular VAE-on-MetroPT3 setup is prone to.
    restored_from = ""
    if (
        train_cfg.early_stopping_restore_best
        and best_auc_encoder_weights is not None
        and best_auc_decoder_weights is not None
        and best_auc_epoch != (history["epoch"][-1] if history["epoch"] else 0)
    ):
        encoder.set_weights(best_auc_encoder_weights)
        decoder.set_weights(best_auc_decoder_weights)
        restored_from = f"best_auc_epoch={best_auc_epoch} (val_roc_auc={best_auc:.4f})"

    early_stop_info["best_epoch"] = int(best_epoch)
    early_stop_info["best_value"] = float(best_metric) if best_metric is not None else float("nan")
    early_stop_info["best_auc_epoch"] = int(best_auc_epoch)
    early_stop_info["best_auc"] = float(best_auc) if best_auc >= 0 else float("nan")
    early_stop_info["restored_from"] = restored_from
    early_stop_info["completed_epochs"] = int(history["epoch"][-1]) if history["epoch"] else 0
    history["early_stopping"] = early_stop_info

    return encoder, decoder, history


def reconstruction_scores(encoder, decoder, windows: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Backward-compatible score API: negative log reconstruction probability.

    Sampling is deterministic during inference because Sampling.call returns the
    latent mean when training=False.
    """
    scores, _ = reconstruction_probability_scores(encoder, decoder, windows, batch_size=batch_size, sigma2=None)
    return scores


def mse_reconstruction_scores(encoder, decoder, windows: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Return per-window reconstruction MSE (higher = more anomalous)."""
    ds = tf.data.Dataset.from_tensor_slices(windows).batch(batch_size)
    scores = []
    for batch_x in ds:
        _, _, z = encoder(batch_x, training=False)
        x_hat = decoder(z, training=False)
        mse = tf.reduce_mean(tf.math.squared_difference(batch_x, x_hat), axis=[1, 2])
        scores.append(mse.numpy())
    return np.concatenate(scores, axis=0) if scores else np.empty((0,), dtype=np.float32)


def reconstruction_probability_scores(
    encoder,
    decoder,
    windows: np.ndarray,
    batch_size: int = 512,
    sigma2: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return negative log reconstruction probability scores and the variance used.

    Lower reconstruction probability corresponds to higher NLL score.
    """
    ds = tf.data.Dataset.from_tensor_slices(windows).batch(batch_size)
    sse_all = []
    for batch_x in ds:
        _, _, z = encoder(batch_x, training=False)
        x_hat = decoder(z, training=False)
        sse = tf.reduce_sum(tf.math.squared_difference(batch_x, x_hat), axis=[1, 2])
        sse_all.append(sse.numpy())

    if not sse_all:
        return np.empty((0,), dtype=np.float32), float("nan")

    sse_all = np.concatenate(sse_all, axis=0)
    d = windows.shape[1] * windows.shape[2]
    sigma2_used = float(np.var(sse_all / max(1, d)) + 1e-8) if sigma2 is None else float(max(sigma2, 1e-8))
    nll = 0.5 * (sse_all / sigma2_used + d * np.log(2.0 * np.pi * sigma2_used))
    return nll.astype(np.float32), sigma2_used


def compute_threshold(train_scores: np.ndarray, cfg: ThresholdConfig) -> float:
    if cfg.method == "mean_std":
        return float(np.mean(train_scores) + cfg.std_factor * np.std(train_scores))
    return float(np.percentile(train_scores, cfg.percentile))


def optimize_threshold_by_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Find threshold maximizing F1 on a labeled validation set."""
    if len(scores) == 0:
        raise ValueError("Validation scores are empty.")

    labels = labels.astype(np.int32)
    if np.unique(labels).size < 2:
        # Degenerate labels; fall back to upper percentile threshold.
        th = float(np.percentile(scores, 95.0))
        return th, 0.0

    qs = np.linspace(0.01, 0.99, 200)
    thresholds = np.unique(np.quantile(scores, qs))

    best_f1 = -1.0
    best_th = float(thresholds[len(thresholds) // 2])
    for th in thresholds:
        pred = (scores > th).astype(np.int32)
        m = binary_metrics(labels, pred)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = float(th)
    return best_th, float(best_f1)


def optimize_threshold_train_percentile(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    val_labels: np.ndarray,
) -> tuple[float, float, float]:
    """Find the train-score percentile threshold that maximizes F1 on the val set.

    Searches over percentiles of the training score distribution (90th–99.9th),
    translates each to an absolute threshold, and evaluates val F1.  This is
    more conservative than searching over val-score quantiles and generalises
    better when val and test anomaly types differ.

    Returns (threshold, best_percentile, best_val_f1).
    """
    labels = val_labels.astype(np.int32)
    if np.unique(labels).size < 2:
        th = float(np.percentile(train_scores, 95.0))
        return th, 95.0, 0.0

    candidate_percentiles = np.linspace(90.0, 99.9, 200)
    candidate_thresholds = np.unique(np.percentile(train_scores, candidate_percentiles))

    best_f1 = -1.0
    best_th = float(candidate_thresholds[len(candidate_thresholds) // 2])
    best_pct = 95.0
    for th in candidate_thresholds:
        pred = (val_scores > th).astype(np.int32)
        m = binary_metrics(labels, pred)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = float(th)
            # record approximate percentile
            best_pct = float(np.mean(train_scores <= best_th) * 100.0)
    return best_th, best_pct, float(best_f1)


def select_threshold(
    train_scores: np.ndarray,
    cfg: ThresholdConfig,
    val_scores: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
) -> tuple[float, dict[str, float | str]]:
    """Select threshold according to configured strategy; supports val F1 maximization."""
    if cfg.method == "val_f1":
        if val_scores is None or val_labels is None:
            raise ValueError("val_scores and val_labels are required when method='val_f1'.")
        th, best_f1 = optimize_threshold_by_f1(val_scores, val_labels)
        return th, {"method": "val_f1", "best_val_f1": best_f1}

    if cfg.method == "train_percentile_val_f1":
        if val_scores is None or val_labels is None:
            raise ValueError("val_scores and val_labels are required when method='train_percentile_val_f1'.")
        th, best_pct, best_f1 = optimize_threshold_train_percentile(train_scores, val_scores, val_labels)
        return th, {"method": "train_percentile_val_f1", "best_train_percentile": best_pct, "best_val_f1": best_f1}

    th = compute_threshold(train_scores, cfg)
    return th, {"method": cfg.method}


def classify(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores > threshold).astype(np.int32)


def find_latest_processed_window_run(processed_windows_root: str | Path, require_val: bool = False) -> Path:
    root = Path(processed_windows_root)
    runs = sorted([p for p in root.iterdir() if p.is_dir()])
    if require_val:
        runs = [p for p in runs if (p / "val_windows.npy").exists() and (p / "val_window_labels.npy").exists()]
    if not runs:
        if require_val:
            raise FileNotFoundError(
                f"No processed window runs with validation files found in {root}. "
                "Run scripts/preprocess.py, then scripts/build_windows.py."
            )
        raise FileNotFoundError(f"No processed window runs found in {root}")
    return runs[-1]


def find_processed_window_run_by_date(
    processed_windows_root: str | Path,
    target_date: str | pd.Timestamp,
    require_val: bool = False,
) -> Path:
    """Find a processed-window run whose saved date ranges cover target_date."""
    root = Path(processed_windows_root)
    target_ts = pd.to_datetime(target_date)
    runs = sorted([p for p in root.iterdir() if p.is_dir()])

    for run_dir in reversed(runs):
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue
        if require_val and not ((run_dir / "val_windows.npy").exists() and (run_dir / "val_window_labels.npy").exists()):
            continue

        metadata = json.loads(meta_path.read_text())
        for period_name in ("train_period", "val_period", "test_period"):
            period = metadata.get(period_name, {})
            start = period.get("start")
            end = period.get("end")
            if start is None or end is None:
                continue
            start_ts = pd.to_datetime(start)
            end_ts = pd.to_datetime(end)
            if start_ts <= target_ts <= end_ts:
                return run_dir

    raise FileNotFoundError(
        f"No processed window run found in {root} containing date {target_ts}."
    )


def find_processed_window_run_by_name(processed_windows_root: str | Path, run_name: str) -> Path:
    """Find a processed-window run by its folder name."""
    run_dir = Path(processed_windows_root) / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"No processed window run found at {run_dir}")
    return run_dir


def load_window_run(run_dir: str | Path):
    run_dir = Path(run_dir)
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {run_dir}")

    metadata = json.loads(meta_path.read_text())
    train_windows = np.load(run_dir / "train_windows.npy")
    val_windows = np.load(run_dir / "val_windows.npy") if (run_dir / "val_windows.npy").exists() else np.empty((0,), dtype=np.float32)
    test_windows = np.load(run_dir / "test_windows.npy")

    train_lbl_path = run_dir / "train_window_labels.npy"
    val_lbl_path = run_dir / "val_window_labels.npy"
    test_lbl_path = run_dir / "test_window_labels.npy"
    train_labels = np.load(train_lbl_path) if train_lbl_path.exists() else np.zeros((len(train_windows),), dtype=np.int32)
    val_labels = np.load(val_lbl_path) if val_lbl_path.exists() else np.zeros((len(val_windows),), dtype=np.int32)
    test_labels = np.load(test_lbl_path) if test_lbl_path.exists() else np.zeros((len(test_windows),), dtype=np.int32)

    # Engineer step may have flattened windows; reshape for model input.
    if train_windows.ndim == 2:
        params = metadata["params"]
        w = 1 if params.get("point_mode", False) else params["window_size"]
        n_features = len(metadata["feature_cols"])
        train_windows = train_windows.reshape((-1, w, n_features))
        val_windows = val_windows.reshape((-1, w, n_features)) if len(val_windows) else val_windows
        test_windows = test_windows.reshape((-1, w, n_features))

    return (
        train_windows.astype(np.float32),
        val_windows.astype(np.float32),
        test_windows.astype(np.float32),
        train_labels.astype(np.int32),
        val_labels.astype(np.int32),
        test_labels.astype(np.int32),
        metadata,
    )


def save_training_artifacts(
    output_root: str | Path,
    encoder,
    decoder,
    history: dict,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    train_preds: np.ndarray,
    test_preds: np.ndarray,
    test_labels: np.ndarray,
    vae_cfg: VAEConfig,
    train_cfg: TrainConfig,
    threshold_cfg: ThresholdConfig,
    source_run_dir: str | Path,
    val_scores: np.ndarray | None = None,
    val_preds: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    train_labels: np.ndarray | None = None,
    threshold_info: dict | None = None,
    scoring_policy: str = "deterministic_latent_mean",
    run_config: dict | None = None,
    source_metadata: dict | None = None,
    git_commit: str | None = None,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    encoder.save(run_dir / "encoder.keras")
    decoder.save(run_dir / "decoder.keras")

    history_table = {k: v for k, v in history.items() if isinstance(v, list)}
    early_stop_info = history.get("early_stopping", {})
    pd.DataFrame(history_table).to_csv(run_dir / "history.csv", index=False)
    if early_stop_info:
        (run_dir / "early_stopping.json").write_text(json.dumps(early_stop_info, indent=2), encoding="utf-8")
    np.save(run_dir / "train_scores.npy", train_scores)
    np.save(run_dir / "test_scores.npy", test_scores)
    np.save(run_dir / "train_predictions.npy", train_preds)
    np.save(run_dir / "test_predictions.npy", test_preds)
    if train_labels is not None:
        np.save(run_dir / "train_labels.npy", train_labels)
    if val_scores is not None:
        np.save(run_dir / "val_scores.npy", val_scores)
    if val_preds is not None:
        np.save(run_dir / "val_predictions.npy", val_preds)
    if val_labels is not None:
        np.save(run_dir / "val_labels.npy", val_labels)
    np.save(run_dir / "test_labels.npy", test_labels)

    metrics = binary_metrics(test_labels, test_preds) if len(test_labels) else {}
    if len(test_labels) and len(test_scores):
        metrics["roc_auc"] = roc_auc_binary(test_labels, test_scores)
    val_metrics = {}
    if val_labels is not None and val_preds is not None and len(val_labels):
        val_metrics = binary_metrics(val_labels, val_preds)
        if val_scores is not None and len(val_scores):
            val_metrics["roc_auc"] = roc_auc_binary(val_labels, val_scores)

    if run_config is not None:
        (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    summary = {
        "run_id": run_id,
        "timestamp": run_id,
        "git_commit": git_commit,
        "source_processed_windows_run": _serialize_path(source_run_dir),
        "source_metadata": source_metadata or {},
        "run_config": run_config or {},
        "vae_config": asdict(vae_cfg),
        "train_config": asdict(train_cfg),
        "threshold_config": asdict(threshold_cfg),
        "threshold_info": threshold_info or {},
        "threshold": float(threshold),
        "scoring_policy": scoring_policy,
        "early_stopping": early_stop_info,
        "train_samples": int(len(train_scores)),
        "val_samples": int(len(val_scores)) if val_scores is not None else 0,
        "test_samples": int(len(test_scores)),
        "validation_metrics": val_metrics,
        "metrics": metrics,
        "saved_files": {
            "encoder": _serialize_path(run_dir / "encoder.keras"),
            "decoder": _serialize_path(run_dir / "decoder.keras"),
            "history": _serialize_path(run_dir / "history.csv"),
            "config": _serialize_path(run_dir / "config.json") if run_config is not None else "",
            "train_scores": _serialize_path(run_dir / "train_scores.npy"),
            "val_scores": _serialize_path(run_dir / "val_scores.npy") if val_scores is not None else "",
            "test_scores": _serialize_path(run_dir / "test_scores.npy"),
            "train_labels": _serialize_path(run_dir / "train_labels.npy") if train_labels is not None else "",
            "val_labels": _serialize_path(run_dir / "val_labels.npy") if val_labels is not None else "",
            "test_labels": _serialize_path(run_dir / "test_labels.npy"),
            "train_predictions": _serialize_path(run_dir / "train_predictions.npy"),
            "val_predictions": _serialize_path(run_dir / "val_predictions.npy") if val_preds is not None else "",
            "test_predictions": _serialize_path(run_dir / "test_predictions.npy"),
        },
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir, summary
