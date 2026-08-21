"""
scan.py

Optuna-driven hyperparameter scan for SPANet, replacing the static
random-sampled config list (run_hparam_scan.py) with adaptive search:
each trial's hyperparameters are chosen based on results from previous
trials (TPE sampler, Optuna's default), rather than every config being
independent. Also supports pruning -- killing a trial early if it's
clearly underperforming relative to other trials at the same point in
training, checked at every checkpoint (every 10 epochs, matching
train.py's own ModelCheckpoint cadence).

Since train.py runs as a separate subprocess (not in-process), pruning
works by polling the run's checkpoint directory while the subprocess is
still running, reporting each new checkpoint's accuracy to Optuna via
trial.report(), and terminating the subprocess if trial.should_prune()
returns True. No changes to train.py itself are needed.

Search space:
  - hidden_dim: continuous int range
  - batch_size: categorical, kept to powers of 2 for GPU efficiency
    (not fully continuous -- a non-power-of-2 batch size can genuinely
    underperform for reasons unrelated to the actual hyperparameter
    search, so this is a deliberate exception to "continuous ranges")
  - dropout, l2_penalty, focal_gamma: continuous float ranges
  - combine_pair_loss: categorical (not numeric)

The exact current baseline is enqueued as trial 0 via study.enqueue_trial,
so it's evaluated under identical conditions (same epoch budget, same
seed) as every adaptively-chosen trial, rather than only existing as a
separate, differently-configured reference run.

Usage:
    python scan_optuna.py --run_config scan_run_config.json --n_trials 20
    # or fully via CLI:
    python scan_optuna.py --baseline HHbbVV_training.json \
        --event_file ... --training_file ... --validation_file ... \
        --log_dir ... --name optuna_scan --epochs 50 --seed 42 --n_trials 20
"""

import argparse
import json
import os
import re
import subprocess
import time

import optuna


CHECKPOINT_PATTERN = re.compile(
    r"epoch=(\d+)-step=(\d+)-validation_average_jet_accuracy=([\d.]+)\.ckpt"
)


def find_run_version_dir(log_dir, run_name, before_versions):
    """Find the version_N directory a just-launched run created, by set
    difference against what existed before -- avoids guessing version
    numbers, which PyTorch Lightning auto-increments and this script
    doesn't control directly."""
    run_base = os.path.join(log_dir, run_name)
    if not os.path.isdir(run_base):
        return None
    current_versions = set(os.listdir(run_base))
    new_versions = current_versions - before_versions
    if len(new_versions) == 1:
        return os.path.join(run_base, new_versions.pop())
    elif len(new_versions) > 1:
        candidates = [os.path.join(run_base, v) for v in new_versions]
        return max(candidates, key=os.path.getmtime)
    return None


def build_run_options(baseline_path, overrides, out_path):
    """Merge baseline JSON with this trial's suggested hyperparameters,
    write the complete merged file. Does NOT mutate the baseline file."""
    with open(baseline_path) as f:
        merged = json.load(f)
    merged.update(overrides)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=4)
    return merged


def run_trial_with_pruning(trial, overrides, args, work_dir):
    """Launches train.py as a non-blocking subprocess, polls its checkpoint
    directory for new results, reports each to Optuna, and prunes (kills
    the subprocess) if Optuna's pruner says this trial isn't competitive.
    Returns the best accuracy seen if the trial completes normally."""
    label = f"trial_{trial.number}"
    merged_path = os.path.join(work_dir, f"{label}.json")
    build_run_options(args.baseline, overrides, merged_path)

    run_base = os.path.join(args.log_dir, args.name)
    before_versions = set(os.listdir(run_base)) if os.path.isdir(run_base) else set()

    cmd = [
        "python", "-m", args.train_module,
        "-ef", args.event_file,
        "-tf", args.training_file,
        "-vf", args.validation_file,
        "-of", merged_path,
        "-l", args.log_dir,
        "-n", args.name,
        "-e", str(args.epochs),
        "-r", str(args.seed),
    ]
    if "batch_size" in overrides:
        cmd.extend(["-b", str(overrides["batch_size"])])

    print(f"\n[Trial {trial.number}] Overrides: {overrides}")
    print(f"[Trial {trial.number}] Running: {' '.join(cmd)}")

    log_path = os.path.join(work_dir, f"{label}.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    version_dir = None
    for _ in range(args.startup_timeout * 4):
        if proc.poll() is not None:
            # Process already exited before ever creating a version dir --
            # almost certainly a crash, not a slow cold start. Fail fast
            # with the real exit code and point at the captured log,
            # rather than waiting out the full timeout for something that
            # will never appear.
            log_file.close()
            raise RuntimeError(
                f"[Trial {trial.number}] train.py exited (code {proc.returncode}) before creating a "
                f"version directory -- see {log_path} for the actual error output."
            )
        version_dir = find_run_version_dir(args.log_dir, args.name, before_versions)
        if version_dir:
            break
        time.sleep(0.25)
    if version_dir is None:
        proc.terminate()
        proc.wait()
        log_file.close()
        raise RuntimeError(
            f"[Trial {trial.number}] version directory never appeared after {args.startup_timeout}s "
            f"(process still running, so this looks like a genuinely slow cold start, not a crash -- "
            f"e.g. dataset loading / feature normalization can take a while) -- see {log_path} for what "
            f"it was doing. Consider raising --startup_timeout if this keeps happening."
        )

    ckpt_dir = os.path.join(version_dir, "checkpoints")
    seen_epochs = set()
    best_acc = None

    def scan_for_new_checkpoints():
        nonlocal best_acc
        if not os.path.isdir(ckpt_dir):
            return
        for fname in os.listdir(ckpt_dir):
            m = CHECKPOINT_PATTERN.match(fname)
            if not m:
                continue
            epoch, step, acc = int(m.group(1)), int(m.group(2)), float(m.group(3))
            if epoch in seen_epochs:
                continue
            seen_epochs.add(epoch)
            if best_acc is None or acc > best_acc:
                best_acc = acc
            print(f"[Trial {trial.number}] epoch={epoch}, acc={acc:.4f}")
            trial.report(acc, epoch)
            if trial.should_prune():
                print(f"[Trial {trial.number}] PRUNED at epoch {epoch} (acc={acc:.4f})")
                proc.terminate()
                proc.wait()
                log_file.close()
                raise optuna.TrialPruned()

    while proc.poll() is None:
        scan_for_new_checkpoints()
        time.sleep(args.poll_interval)

    proc.wait()
    # One final scan AFTER the process has fully exited -- the loop above
    # exits the instant proc.poll() goes non-None, which can race against
    # the LAST checkpoint (e.g. the final epoch) still being flushed to
    # disk right as the process finishes. Without this, every trial's
    # reported accuracy could silently be missing its actual best/final
    # result.
    scan_for_new_checkpoints()
    log_file.close()
    if proc.returncode != 0:
        raise RuntimeError(f"[Trial {trial.number}] train.py exited with code {proc.returncode} -- see {log_path}")

    return best_acc if best_acc is not None else 0.0


def make_objective(args, work_dir):
    def objective(trial):
        overrides = {
            "hidden_dim": trial.suggest_int("hidden_dim", 64, 256, step=32),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.4),
            "l2_penalty": trial.suggest_float("l2_penalty", 1e-5, 1e-3, log=True),
            "combine_pair_loss": trial.suggest_categorical("combine_pair_loss", ["min", "softmin", "mean"]),
            "focal_gamma": trial.suggest_float("focal_gamma", 0.0, 3.0),
        }
        return run_trial_with_pruning(trial, overrides, args, work_dir)
    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_config", default=None,
                         help="JSON file supplying any of the other arguments below. "
                              "Explicit CLI flags always override whatever's in this file.")
    parser.add_argument("--baseline", default=None, help="Baseline options JSON -- each trial's suggested hyperparameters are merged on top of this")
    parser.add_argument("--event_file", default=None)
    parser.add_argument("--training_file", default=None)
    parser.add_argument("--validation_file", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--name", default=None, help="Defaults to 'optuna_scan' if not given via CLI or --run_config")
    parser.add_argument("--epochs", type=int, default=None, help="Defaults to 50")
    parser.add_argument("--seed", type=int, default=None, help="Fixed across every trial. Defaults to 42")
    parser.add_argument("--train_module", default=None, help="Defaults to 'spanet.train'")
    parser.add_argument("--work_dir", default=None, help="Defaults to './optuna_scan_configs'")
    parser.add_argument("--n_trials", type=int, default=None, help="Defaults to 18 (17 adaptive + the enqueued baseline)")
    parser.add_argument("--poll_interval", type=float, default=30.0, help="Seconds between checkpoint-directory polls while a trial is running")
    parser.add_argument("--startup_timeout", type=int, default=300, help="Seconds to wait for a trial's version directory to appear before giving up. Dataset loading + feature normalization can genuinely take a while on a cold start -- this is deliberately generous, and now that stdout/stderr are captured to a log file, a real crash gets detected and reported immediately regardless of this value, rather than waiting out the full timeout.")
    parser.add_argument("--study_db", default=None, help="If set, path to a SQLite file for persisting the study (allows resuming across sessions). Otherwise the study exists only in memory for this run.")
    args = parser.parse_args()

    if args.run_config:
        with open(args.run_config) as f:
            run_cfg = json.load(f)
        for key, value in run_cfg.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)

    if args.name is None:
        args.name = "optuna_scan"
    if args.epochs is None:
        args.epochs = 50
    if args.seed is None:
        args.seed = 42
    if args.train_module is None:
        args.train_module = "spanet.train"
    if args.work_dir is None:
        args.work_dir = "./optuna_scan_configs"
    if args.n_trials is None:
        args.n_trials = 18

    required_fields = ["baseline", "event_file", "training_file", "validation_file", "log_dir"]
    missing = [f for f in required_fields if getattr(args, f) is None]
    if missing:
        raise SystemExit(f"Missing required argument(s), not provided via CLI flag or --run_config: {missing}")

    os.makedirs(args.work_dir, exist_ok=True)

    if args.study_db:
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=0),
            storage=f"sqlite:///{args.study_db}",
            study_name=args.name,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=0))

    # Enqueue the exact current baseline as an explicit trial, evaluated
    # under IDENTICAL conditions (same epoch budget, same seed) as every
    # adaptively-suggested trial -- not just a separately-run reference.
    with open(args.baseline) as f:
        baseline_values = json.load(f)
    baseline_trial_params = {
        "hidden_dim": baseline_values.get("hidden_dim", 128),
        "batch_size": baseline_values.get("batch_size", 256),
        "dropout": baseline_values.get("dropout", 0.2),
        "l2_penalty": baseline_values.get("l2_penalty", 0.0002),
        "combine_pair_loss": baseline_values.get("combine_pair_loss", "softmin"),
        "focal_gamma": baseline_values.get("focal_gamma", 1.0),
    }
    study.enqueue_trial(baseline_trial_params)

    objective = make_objective(args, args.work_dir)
    study.optimize(objective, n_trials=args.n_trials)

    print("\n" + "=" * 90)
    print("SUMMARY: Optuna hyperparameter scan results")
    print("=" * 90)
    print(f"{'Trial':<8} {'State':<10} {'Best Acc':>10}  Params")
    print("-" * 90)
    for t in study.trials:
        acc_str = f"{t.value:.4f}" if t.value is not None else "N/A"
        print(f"{t.number:<8} {t.state.name:<10} {acc_str:>10}  {t.params}")
    print("-" * 90)
    if study.best_trial is not None:
        print(f"Best trial: #{study.best_trial.number}, accuracy={study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
    print("=" * 90)

    return study


if __name__ == "__main__":
    main()
