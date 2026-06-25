"""
Optuna tuning script for 2024-CGAD that saves the best training command.

Put this file in the project root of https://github.com/weiranli-cjlu/2024-CGAD
and run, for example:

    pip install optuna scikit-learn
    python optuna_tune_cgad_save_command.py \
        --dataset cora \
        --data_dir ~/datasets/GAD/mat \
        --device cuda:0 \
        --n_trials 50 \
        --num_epoch 100 \
        --auc_test_rounds 50 \
        --quiet

Outputs under --output_dir:
- <study_name>.db                         Optuna sqlite study
- <study_name>_trials.csv                 all trials and metrics
- <study_name>_best.json                  best params, metrics, command
- <study_name>_best_train_command.txt     single-line reproducible command
- <study_name>_best_train_command.sh      executable shell script

Notes:
- AUPRC is computed with sklearn.metrics.precision_recall_curve + sklearn.metrics.auc.
- Trials with NaN/Inf y_score are pruned by default so tuning continues.
- The saved command calls the repository's main.py using the best hyperparameters.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import optuna
import torch
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import precision_recall_curve, roc_auc_score

# Make imports work when this file is placed in the CGAD project root.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run.run import train_ours  # noqa: E402


class NonFiniteScoreError(ValueError):
    """Raised when y_score cannot be used to compute valid metrics."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna tuner for CGAD; saves best python main.py training command"
    )

    # Fixed CGAD experiment options.
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data_dir", type=str, default="~/datasets/GAD/mat")
    parser.add_argument(
        "--train_dir",
        type=str,
        default="./runs",
        help="Training directory that will be written into the saved best command.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=(
            "Preprocess cache directory. If omitted, the saved command also omits --cache_dir "
            "so current main.py saves cache under <train_dir>/cgad_preprocess."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1, help="CGAD internal runs per Optuna trial and saved command")
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--auc_test_rounds", type=int, default=100)
    parser.add_argument("--community_method", type=str, default="louvain", choices=["louvain", "greedy", "components"])
    parser.add_argument("--max_communities", type=int, default=0, help="0 means keep all generated communities")
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="suppress train_ours stdout logs during each trial")

    # Saved command options.
    parser.add_argument(
        "--best_command_prefix",
        type=str,
        default="python main.py",
        help="Command prefix used when saving the final best training instruction.",
    )
    parser.add_argument(
        "--best_results_csv",
        type=str,
        default=None,
        help=(
            "results_csv path written into the saved best command. "
            "Default: <output_dir>/<study_name>_best_train_results.csv"
        ),
    )
    parser.add_argument(
        "--save_score_run",
        type=int,
        default=-1,
        help="Passed to saved best command; <=0 means do not save y_true/y_score.",
    )
    parser.add_argument(
        "--score_save_dir",
        type=str,
        default=None,
        help="Passed to saved best command when --save_score_run > 0.",
    )

    # NaN/Inf handling.
    parser.add_argument(
        "--nan_policy",
        type=str,
        default="prune",
        choices=["prune", "replace"],
        help=(
            "prune: prune any trial whose y_score contains NaN/Inf; "
            "replace: replace partial NaN/Inf with finite min/max, but still prune if all scores are non-finite."
        ),
    )
    parser.add_argument(
        "--min_finite_score_ratio",
        type=float,
        default=1.0,
        help=(
            "Minimum finite ratio required when --nan_policy replace is used. "
            "Default 1.0 means any NaN/Inf trial is pruned unless you lower this value."
        ),
    )
    parser.add_argument(
        "--max_score_abs",
        type=float,
        default=1e12,
        help="Prune trial if absolute finite score is larger than this threshold.",
    )

    # Optuna options.
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None, help="seconds; omit for no timeout")
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None, help="e.g. sqlite:///optuna_cgad.db; default uses output_dir")
    parser.add_argument("--output_dir", type=str, default="optuna_results")
    parser.add_argument("--objective_metric", type=str, default="auc", choices=["auprc", "auc", "mean"])
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--enqueue_default", action="store_true", help="evaluate current main.py defaults as the first trial")

    # Search-space controls. Defaults are conservative to reduce exp/logit overflow.
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--max_lr", type=float, default=5e-3)
    parser.add_argument("--embedding_dims", type=str, default="32,48,64,96,128")
    parser.add_argument("--batch_sizes", type=str, default="64,128,256,512")
    parser.add_argument("--min_subgraph_size", type=int, default=3)
    parser.add_argument("--max_subgraph_size", type=int, default=8)
    parser.add_argument("--min_T", type=float, default=0.5)
    parser.add_argument("--max_T", type=float, default=2.0)
    parser.add_argument("--min_q", type=float, default=0.1)
    parser.add_argument("--max_q", type=float, default=0.7)
    parser.add_argument("--min_lam", type=float, default=0.1)
    parser.add_argument("--max_lam", type=float, default=0.8)

    return parser.parse_args()


def parse_int_list(text: str) -> List[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError(f"empty integer list: {text!r}")
    return values


@contextlib.contextmanager
def maybe_suppress_stdout(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def build_cgad_args(cli: argparse.Namespace, trial: optuna.trial.Trial) -> SimpleNamespace:
    """Build the args object expected by run.run.train_ours."""
    embedding_dims = parse_int_list(cli.embedding_dims)
    batch_sizes = parse_int_list(cli.batch_sizes)

    args = SimpleNamespace()

    # Basic experiment settings used by train_ours / utils.load_mat_data.
    args.expid = trial.number
    args.device = cli.device
    args.runs = cli.runs
    args.seed = cli.seed + trial.number * 1000
    args.dataset = cli.dataset
    args.data_dir = cli.data_dir
    args.train_dir = cli.train_dir
    args.cache_dir = cli.cache_dir
    args.force_preprocess = cli.force_preprocess
    args.community_method = cli.community_method
    args.max_communities = cli.max_communities

    # Fixed training budget.
    args.num_epoch = cli.num_epoch
    args.patience = cli.patience
    args.auc_test_rounds = cli.auc_test_rounds

    # Keep compatibility with main.py / model.Model.
    args.num_community = 3
    args.grid_search = False
    args.loss_fun = "rnce"

    # Tuned hyperparameters.
    args.lr = trial.suggest_float("lr", cli.min_lr, cli.max_lr, log=True)
    args.weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-8, 1e-6, 1e-5, 1e-4])
    args.embedding_dim = trial.suggest_categorical("embedding_dim", embedding_dims)
    args.batch_size = trial.suggest_categorical("batch_size", batch_sizes)
    args.subgraph_size = trial.suggest_int("subgraph_size", cli.min_subgraph_size, cli.max_subgraph_size)
    args.readout = trial.suggest_categorical("readout", ["avg", "max", "weighted_sum"])
    args.neg_sample_method = trial.suggest_categorical("neg_sample_method", ["bias", "even", "random"])
    args.num_negs = trial.suggest_int("num_negs", 1, 8)
    args.strategy = trial.suggest_categorical("strategy", ["most-relevant", "least-relevant", "random"])
    args.alpha = trial.suggest_float("alpha", 0.1, 0.9, step=0.1)
    args.lam = trial.suggest_float("lam", cli.min_lam, cli.max_lam)
    args.T = trial.suggest_float("T", cli.min_T, cli.max_T)
    args.q = trial.suggest_float("q", cli.min_q, cli.max_q)

    args.tqdm = False

    return args


def score_diagnostics(y_score: np.ndarray) -> Dict[str, Any]:
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(y_score)
    finite_values = y_score[finite_mask]
    diag: Dict[str, Any] = {
        "score_size": int(y_score.size),
        "score_nan_count": int(np.isnan(y_score).sum()),
        "score_posinf_count": int(np.isposinf(y_score).sum()),
        "score_neginf_count": int(np.isneginf(y_score).sum()),
        "score_finite_count": int(finite_mask.sum()),
        "score_finite_ratio": float(finite_mask.mean()) if y_score.size else 0.0,
    }
    if finite_values.size:
        diag.update(
            {
                "score_min": float(np.min(finite_values)),
                "score_max": float(np.max(finite_values)),
                "score_mean": float(np.mean(finite_values)),
                "score_std": float(np.std(finite_values)),
            }
        )
    else:
        diag.update({"score_min": None, "score_max": None, "score_mean": None, "score_std": None})
    return diag


def sanitize_y_score(
    y_score: Iterable[float],
    cli: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Validate y_score and optionally replace partial NaN/Inf values."""
    y_score_arr = np.asarray(y_score, dtype=np.float64).reshape(-1)
    diag = score_diagnostics(y_score_arr)

    if y_score_arr.size == 0:
        raise NonFiniteScoreError("empty y_score; cannot compute metrics")

    finite_mask = np.isfinite(y_score_arr)
    finite_count = int(finite_mask.sum())

    if finite_count == 0:
        raise NonFiniteScoreError(
            "all values in y_score are NaN/Inf; this usually means the trial diverged during training"
        )

    finite_values = y_score_arr[finite_mask]
    max_abs = float(np.max(np.abs(finite_values)))
    if max_abs > cli.max_score_abs:
        raise NonFiniteScoreError(
            f"finite y_score values are numerically unstable: max_abs={max_abs:.6e} > {cli.max_score_abs:.6e}"
        )

    if float(diag["score_finite_ratio"]) < cli.min_finite_score_ratio:
        raise NonFiniteScoreError(
            "finite y_score ratio is too low: "
            f"{diag['score_finite_ratio']:.6f} < {cli.min_finite_score_ratio:.6f}"
        )

    if finite_count < y_score_arr.size:
        if cli.nan_policy == "prune":
            raise NonFiniteScoreError(
                "y_score contains NaN/Inf; prune this unstable hyperparameter combination"
            )

        # Conservative replacement for partial non-finite values only.
        finite_min = float(np.min(finite_values))
        finite_max = float(np.max(finite_values))
        y_score_arr = np.nan_to_num(
            y_score_arr,
            nan=finite_min,
            posinf=finite_max,
            neginf=finite_min,
        )
        diag["score_nan_replaced"] = True
    else:
        diag["score_nan_replaced"] = False

    if float(np.std(y_score_arr)) == 0.0:
        raise NonFiniteScoreError("all finite y_score values are identical; metrics are uninformative")

    return y_score_arr, diag


def compute_metrics(y_true: Iterable[int], y_score: Iterable[float], cli: argparse.Namespace) -> Tuple[Dict[str, float], Dict[str, Any]]:
    y_true_arr = np.asarray(y_true).astype(int).reshape(-1)
    y_score_arr, diag = sanitize_y_score(y_score, cli)

    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError(f"length mismatch: len(y_true)={y_true_arr.shape[0]}, len(y_score)={y_score_arr.shape[0]}")

    if len(np.unique(y_true_arr)) < 2:
        raise ValueError("y_true must contain both normal and anomaly classes to compute AUC/AUPRC.")

    auc_value = float(roc_auc_score(y_true_arr, y_score_arr))

    # precision_recall_curve usually returns recall in descending order;
    # reverse before trapezoidal auc so the x-axis is increasing.
    precision, recall, _ = precision_recall_curve(y_true_arr, y_score_arr)
    auprc_value = float(sklearn_auc(recall[::-1], precision[::-1]))

    k = max(int(y_true_arr.sum()), 1)
    topk = np.argsort(-y_score_arr)[:k]
    recall_at_k = float(y_true_arr[topk].sum() / max(y_true_arr.sum(), 1))

    metrics = {
        "auc": auc_value,
        "auprc": auprc_value,
        "recall_at_k": recall_at_k,
    }
    return metrics, diag


def objective_value(metrics: Dict[str, float], objective_metric: str) -> float:
    if objective_metric == "auc":
        value = metrics["auc"]
    elif objective_metric == "mean":
        value = (metrics["auc"] + metrics["auprc"]) / 2.0
    else:
        value = metrics["auprc"]
    if not np.isfinite(value):
        raise NonFiniteScoreError(f"objective value is not finite: {value}")
    return float(value)


def flatten_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {f"param_{k}": v for k, v in params.items()}


def shell_join(tokens: List[Any]) -> str:
    return " ".join(shlex.quote(str(token)) for token in tokens)


def build_best_training_command(
    best_params: Dict[str, Any],
    cli: argparse.Namespace,
) -> str:
    """Build a reproducible command that calls the repository main.py with best params."""
    tokens: List[Any] = shlex.split(cli.best_command_prefix)

    # Fixed options from current main.py.
    tokens += [
        "--dataset", cli.dataset,
        "--data_dir", cli.data_dir,
        "--train_dir", cli.train_dir,
        "--device", cli.device,
        "--runs", 10,
        "--seed", cli.seed,
        "--num_epoch", cli.num_epoch,
        "--patience", cli.patience,
        "--auc_test_rounds", cli.auc_test_rounds,
        "--community_method", cli.community_method,
        "--max_communities", cli.max_communities,
    ]

    if cli.cache_dir is not None:
        tokens += ["--cache_dir", cli.cache_dir]
    if cli.force_preprocess:
        tokens += ["--force_preprocess"]
    if cli.save_score_run > 0:
        tokens += ["--save_score_run", cli.save_score_run]
        if cli.score_save_dir is not None:
            tokens += ["--score_save_dir", cli.score_save_dir]

    # Best Optuna parameters that are accepted by main.py.
    param_order = [
        "lr",
        "weight_decay",
        "embedding_dim",
        "batch_size",
        "subgraph_size",
        "readout",
        "neg_sample_method",
        "num_negs",
        "strategy",
        "alpha",
        "lam",
        "T",
        "q",
    ]
    for key in param_order:
        if key in best_params:
            tokens += [f"--{key}", best_params[key]]

    # main.py keeps this argument for compatibility.
    tokens += ["--loss_fun", "rnce", "--tqdm"]

    return shell_join(tokens)


def save_best_command_files(
    study: optuna.Study,
    cli: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, str]:
    """Save best training command as .txt and executable .sh."""
    try:
        best = study.best_trial
    except ValueError:
        return {}

    command = build_best_training_command(best.params, cli)

    sh_path = output_dir / f"{cli.study_name}_best_train_command.sh"

    sh_path.write_text(command + "\n", encoding="utf-8")
    sh_path.chmod(0o755)

    return {
        "best_train_command": command,
        "best_train_command_sh": str(sh_path),
    }


def export_trials_csv(study: optuna.Study, csv_path: Path, cli: argparse.Namespace) -> None:
    """Export all trials without requiring pandas."""
    trials = list(study.trials)
    param_keys = sorted({key for t in trials for key in t.params.keys()})
    user_attr_keys = sorted({key for t in trials for key in t.user_attrs.keys()})

    fieldnames = [
        "datetime",
        "dataset",
        "trial",
        "state",
        "objective_metric",
        "value",
        "auc",
        "auprc",
        "recall_at_k",
        "auc_percent",
        "auprc_percent",
        "duration_seconds",
    ]
    fieldnames += [f"param_{k}" for k in param_keys]
    fieldnames += [f"user_{k}" for k in user_attr_keys if k not in {"auc", "auprc", "recall_at_k"}]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trials:
            auc_value = t.user_attrs.get("auc", "")
            auprc_value = t.user_attrs.get("auprc", "")
            row: Dict[str, Any] = {
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "dataset": cli.dataset,
                "trial": t.number,
                "state": t.state.name,
                "objective_metric": cli.objective_metric,
                "value": "" if t.value is None else t.value,
                "auc": auc_value,
                "auprc": auprc_value,
                "recall_at_k": t.user_attrs.get("recall_at_k", ""),
                "auc_percent": "" if auc_value == "" else f"{float(auc_value) * 100:.2f}",
                "auprc_percent": "" if auprc_value == "" else f"{float(auprc_value) * 100:.2f}",
                "duration_seconds": "" if t.duration is None else round(t.duration.total_seconds(), 3),
            }
            row.update(flatten_params(t.params))
            for key in user_attr_keys:
                if key in {"auc", "auprc", "recall_at_k"}:
                    continue
                row[f"user_{key}"] = t.user_attrs.get(key, "")
            writer.writerow(row)


def save_best_json(study: optuna.Study, path: Path, cli: argparse.Namespace, command_payload: Dict[str, str]) -> None:
    try:
        best = study.best_trial
    except ValueError:
        return
    payload = {
        "dataset": cli.dataset,
        "objective_metric": cli.objective_metric,
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "best_train_command": command_payload.get("best_train_command"),
        "best_train_command_sh": command_payload.get("best_train_command_sh"),
        "best_train_results_csv": command_payload.get("best_train_results_csv"),
        "metrics": {
            "auc": best.user_attrs.get("auc"),
            "auprc": best.user_attrs.get("auprc"),
            "recall_at_k": best.user_attrs.get("recall_at_k"),
        },
        "fixed_args": {
            "data_dir": cli.data_dir,
            "train_dir": cli.train_dir,
            "cache_dir": cli.cache_dir,
            "device": cli.device,
            "runs": cli.runs,
            "seed": cli.seed,
            "num_epoch": cli.num_epoch,
            "patience": cli.patience,
            "auc_test_rounds": cli.auc_test_rounds,
            "community_method": cli.community_method,
            "max_communities": cli.max_communities,
            "force_preprocess": cli.force_preprocess,
            "nan_policy": cli.nan_policy,
            "min_finite_score_ratio": cli.min_finite_score_ratio,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _set_trial_attrs(trial: optuna.trial.Trial, attrs: Dict[str, Any]) -> None:
    for key, value in attrs.items():
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        trial.set_user_attr(key, value)


def main() -> None:
    cli = parse_args()
    output_dir = Path(cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if cli.study_name is None:
        cli.study_name = f"cgad_{cli.dataset}_{cli.objective_metric}"
    if cli.storage is None:
        cli.storage = f"sqlite:///{output_dir / (cli.study_name + '.db')}"

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    sampler = optuna.samplers.TPESampler(seed=cli.sampler_seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    study = optuna.create_study(
        study_name=cli.study_name,
        storage=cli.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    if cli.enqueue_default:
        default_params = {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "embedding_dim": 64,
            "batch_size": 256,
            "subgraph_size": 4,
            "readout": "avg",
            "neg_sample_method": "bias",
            "num_negs": 3,
            "strategy": "most-relevant",
            "alpha": 0.5,
            "lam": 0.5,
            "T": 1.0,
            "q": 0.3,
        }
        try:
            study.enqueue_trial(default_params, skip_if_exists=True)
        except TypeError:
            study.enqueue_trial(default_params)

    trials_csv = output_dir / f"{cli.study_name}_trials.csv"
    best_json = output_dir / f"{cli.study_name}_best.json"

    def objective(trial: optuna.trial.Trial) -> float:
        args = build_cgad_args(cli, trial)
        try:
            with maybe_suppress_stdout(cli.quiet):
                y_true, y_score = train_ours(args)

            metrics, diag = compute_metrics(y_true, y_score, cli)
            value = objective_value(metrics, cli.objective_metric)

            _set_trial_attrs(trial, metrics)
            _set_trial_attrs(trial, diag)
            trial.set_user_attr("seed", args.seed)
            trial.set_user_attr("error_type", "")
            trial.report(value, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return value

        except NonFiniteScoreError as exc:
            try:
                if "y_score" in locals():
                    _set_trial_attrs(trial, score_diagnostics(np.asarray(y_score, dtype=np.float64)))
            except Exception:
                pass
            trial.set_user_attr("error_type", "non_finite_score")
            trial.set_user_attr("error", str(exc)[:500])
            raise optuna.TrialPruned() from exc

        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda" in msg:
                trial.set_user_attr("error_type", "runtime_cuda_or_oom")
                trial.set_user_attr("error", str(exc)[:500])
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned() from exc
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def callback(study_obj: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        export_trials_csv(study_obj, trials_csv, cli)
        command_payload = save_best_command_files(study_obj, cli, output_dir)
        save_best_json(study_obj, best_json, cli, command_payload)
        if trial.state == optuna.trial.TrialState.COMPLETE:
            auc_v = trial.user_attrs.get("auc", float("nan"))
            auprc_v = trial.user_attrs.get("auprc", float("nan"))
            print(
                f"Trial {trial.number} finished: value={trial.value:.6f}, "
                f"AUC={auc_v:.6f}, AUPRC={auprc_v:.6f}, params={trial.params}",
                flush=True,
            )
        else:
            err_type = trial.user_attrs.get("error_type", "")
            err = trial.user_attrs.get("error", "")
            if err_type:
                print(f"Trial {trial.number} ended with state={trial.state.name}, {err_type}: {err}", flush=True)
            else:
                print(f"Trial {trial.number} ended with state={trial.state.name}", flush=True)

    study.optimize(objective, n_trials=cli.n_trials, timeout=cli.timeout, callbacks=[callback], gc_after_trial=True, show_progress_bar=True)

    export_trials_csv(study, trials_csv, cli)
    command_payload = save_best_command_files(study, cli, output_dir)
    save_best_json(study, best_json, cli, command_payload)

    print("\n========== Optuna tuning finished ==========")
    print(f"Study name: {cli.study_name}")
    print(f"Storage: {cli.storage}")
    print(f"Trials CSV: {trials_csv}")
    print(f"Best JSON: {best_json}")
    try:
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value ({cli.objective_metric}): {study.best_value:.6f}")
        print("Best params:")
        print(json.dumps(study.best_params, ensure_ascii=False, indent=2))
        if command_payload:
            print("Best training command:")
            print(command_payload["best_train_command"])
            print(f"Best command sh: {command_payload['best_train_command_sh']}")
    except ValueError:
        print(
            "No completed trial. Most trials may have produced NaN/Inf scores. "
            "Try reducing --max_lr, --max_q, --max_lam or increasing --min_T."
        )


if __name__ == "__main__":
    main()
