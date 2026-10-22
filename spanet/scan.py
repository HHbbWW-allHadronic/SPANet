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
still running, reporting each new periodic checkpoint's validation_significance to Optuna via
trial.report(), and terminating the subprocess if trial.should_prune()
returns True. No changes to train.py itself are needed.

Search space (see ADJUST_PARAMS below -- the single place it is defined):
  - learning_rate (log), dropout, l2_penalty (log), focal_gamma: continuous
    float ranges
  - batch_size, hidden_dim: categorical (batch_size kept to powers of 2 for
    GPU efficiency; hidden_dim kept to values divisible by the attention
    head count of the baseline)
  - classification_loss_scale (log): continuous float range -- how strongly
    the classification head competes with assignment/detection for the
    shared encoder
  Everything else (encoder/branch depths, attention heads, combine_pair_loss,
  assignment/detection loss scales) is taken unchanged from the baseline JSON.

The exact current baseline is enqueued as trial 0 via study.enqueue_trial,
so it's evaluated under identical conditions (same epoch budget, same
seed) as every adaptively-chosen trial, rather than only existing as a
separate, differently-configured reference run.

Usage:
    python scan_optuna.py --run_config scan_run_config.json --n_trials 20
    # or fully via CLI, same as run_hparam_scan.py:
    python scan_optuna.py --baseline HHbbVV_training.json \
        --event_file ... --training_file ... --validation_file ... \
        --log_dir ... --name optuna_scan --epochs 50 --seed 42 --n_trials 20
"""

import argparse
import bisect
import glob
import json
import os
import re
import subprocess
import time
from os import urandom
from datetime import datetime

import optuna


# train.py now saves two kinds of checkpoints, both named with validation_significance (weighted binned Z from the
# detection scores, logged by JetReconstructionValidation):
#   periodic: epoch=9-step=12690-validation_significance=0.04012.ckpt       (every 10 epochs, all kept)
#   best:     best-epoch=3-step=5076-validation_significance=0.04512.ckpt   (top 3 by significance, written every epoch)
CHECKPOINT_PATTERN = re.compile(
    r"(best-)?epoch=(\d+)-step=(\d+)-validation_(?:significance|sensitivity)=([\d.]+)\.ckpt"
)

# A trial's score is the MEAN of its TOP_K_EPOCHS best per-epoch values, not the single best epoch: epoch-to-epoch
# (and seed-to-seed) noise in the metric is ~10%, and a single max rewards a lucky epoch.
TOP_K_EPOCHS = 3


def parse_checkpoint(fname):
    """(is_best_file, epoch, step, significance) for a checkpoint filename, or None if it is not one of ours."""
    m = CHECKPOINT_PATTERN.match(fname)
    if not m:
        return None
    return bool(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))


def trial_score(epoch_values, k=TOP_K_EPOCHS):
    """Mean of the k highest per-epoch values ({epoch: value}); fewer than k epochs -> mean of what exists."""
    if not epoch_values:
        return None
    top = sorted(epoch_values.values(), reverse=True)[:k]
    return sum(top) / len(top)


def generate_seed() -> int:
    """Draw a fresh seed from OS entropy, matching train.py's own
    generate_seed() -- same mechanism, same range (0 to 2**32-1), so a
    seed printed by either tool is safe to hand to the other."""
    return int.from_bytes(urandom(4), "big")


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


def _nearest_value(sorted_steps, sorted_values, target_step):
    """Return the value whose step is closest to target_step, via bisect
    on the (already step-sorted, as TensorBoard writes them) list --
    used to pair up loss and learning-rate scalars that were logged at
    slightly different steps (e.g. loss per training step, LR only when
    the scheduler actually changes it) rather than requiring an exact
    step match, which would silently drop most points."""
    idx = bisect.bisect_left(sorted_steps, target_step)
    candidates = []
    if idx < len(sorted_steps):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    best_idx = min(candidates, key=lambda i: abs(sorted_steps[i] - target_step))
    return sorted_values[best_idx]


def save_loss_vs_lr_plot(version_dir, output_path, trial_number=None,
                          loss_tag_hint=None, lr_tag_hint=None):
    """For one completed trial's TensorBoard run, plot training loss
    against the actual instantaneous learning rate logged at each step
    -- NOT the trial's fixed base learning_rate hyperparameter. Warmup
    plus the cosine (possibly hard-restart) scheduler mean the LR moves
    throughout a single run, so this traces how loss responds as LR
    sweeps through its schedule -- points are colored by step so the
    (non-monotonic, once cycles restart) direction of travel is
    visible.

    Loss/LR tag names are auto-detected by case-insensitive substring
    match, since the exact tag names PyTorch Lightning writes depend on
    exactly which self.log(...) calls / callbacks SPANet's training
    module uses, which this script doesn't otherwise inspect. Pass
    loss_tag_hint/lr_tag_hint (or --loss_tag/--lr_tag on the CLI) to
    override if the auto-pick guesses wrong -- the tags actually chosen
    are always printed, so a bad guess is visible immediately rather
    than silently producing a meaningless plot.

    Never raises: a missing tag, missing matplotlib, or any other
    failure here is logged and skipped rather than failing the
    trial/scan, since this is a diagnostic extra, not part of the
    actual optimization.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print(f"[loss-vs-lr] tensorboard not available -- skipped for {version_dir}")
        return

    event_files = glob.glob(os.path.join(version_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"[loss-vs-lr] no event file found in {version_dir} -- skipped")
        return

    try:
        ea = EventAccumulator(event_files[0])
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
    except Exception as e:
        print(f"[loss-vs-lr] could not read event file in {version_dir}: {e} -- skipped")
        return

    def pick_tag(hint, keyword_sets, exclude=()):
        if hint:
            return hint if hint in tags else None
        for keywords in keyword_sets:
            candidates = [
                t for t in tags
                if all(k in t.lower() for k in keywords)
                and not any(x in t.lower() for x in exclude)
            ]
            if candidates:
                # Shortest matching name wins -- e.g. plain "loss" over
                # "assignment_loss" -- as the most likely "the" primary
                # metric, while still finding something if there's no
                # exact "loss"/"lr" tag.
                return min(candidates, key=len)
        return None

    lr_tag = pick_tag(lr_tag_hint, [["lr"], ["learning_rate"], ["learning-rate"]])
    loss_tag = pick_tag(loss_tag_hint, [["loss"]], exclude=["val", "validation"])

    if lr_tag is None or loss_tag is None:
        print(f"[loss-vs-lr] couldn't find both a learning-rate tag and a loss tag among: {tags} "
              f"-- skipped (pass --lr_tag/--loss_tag explicitly if these names look wrong)")
        return

    lr_events = ea.Scalars(lr_tag)
    loss_events = ea.Scalars(loss_tag)
    if not lr_events or not loss_events:
        print(f"[loss-vs-lr] '{lr_tag}' or '{loss_tag}' had no logged points -- skipped")
        return

    lr_steps = [e.step for e in lr_events]
    lr_values = [e.value for e in lr_events]

    matched_lr, matched_loss, matched_step = [], [], []
    for e in loss_events:
        matched_lr.append(_nearest_value(lr_steps, lr_values, e.step))
        matched_loss.append(e.value)
        matched_step.append(e.step)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        sc = ax.scatter(matched_lr, matched_loss, c=matched_step, cmap="viridis", s=10)
        ax.set_xlabel(f"Learning rate ({lr_tag})")
        ax.set_ylabel(f"Loss ({loss_tag})")
        title = "Loss vs. learning rate over training"
        if trial_number is not None:
            title = f"Trial {trial_number} -- " + title
        ax.set_title(title)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Step")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"[loss-vs-lr] saved: {output_path}  (tags used: lr='{lr_tag}', loss='{loss_tag}')")
    except ImportError:
        print("[loss-vs-lr] matplotlib not available -- skipped")


_DATASET_PATH_KEYS = {"event_info_file", "training_file", "validation_file", "testing_file"}


def build_run_options(baseline_path, overrides, out_path):
    """Merge baseline JSON with this trial's suggested hyperparameters,
    write the complete merged file. Does NOT mutate the baseline file.

    Deliberately strips event_info_file/training_file/validation_file/
    testing_file from the result, regardless of whether the baseline JSON
    contains them -- these must ALWAYS come from the -ef/-tf/-vf CLI
    flags (which train.py already applies before this file is layered on
    top). Confirmed bug this fixes: train.py's own update_options() has
    no reason to skip these keys, so if a stale baseline JSON still has
    them, they'd silently overwrite whatever correct CLI-provided paths
    were passed -- exactly what happened once already tonight (region4's
    NO_BJET files getting loaded instead of the intended region2 dataset).
    """
    with open(baseline_path) as f:
        merged = json.load(f)
    for key in _DATASET_PATH_KEYS:
        merged.pop(key, None)
    merged.update(overrides)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=4)
    return merged


def run_trial_with_pruning(trial, overrides, args, work_dir):
    """Launches train.py as a non-blocking subprocess, polls its checkpoint
    directory for new results, reports each to Optuna, and prunes (kills
    the subprocess) if Optuna's pruner says this trial isn't competitive.
    Returns the trial score (mean of the top-3 per-epoch validation_significance values) if the trial completes normally."""
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
        "-ds", str(args.seed),     # train.py's dataset seed is separate from -r now; fix it too so trials stay comparable
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
    reported_epochs = set()   # periodic checkpoints already reported to Optuna (pruning checks)
    epoch_values = {}         # {epoch: validation_significance} from ANY checkpoint file seen (periodic or best-)

    def scan_for_new_checkpoints():
        if not os.path.isdir(ckpt_dir):
            return
        for fname in os.listdir(ckpt_dir):
            parsed = parse_checkpoint(fname)
            if parsed is None:
                continue
            is_best, epoch, step, value = parsed
            # Remember every epoch's value as soon as its file appears: the best- files are deleted again once three
            # better epochs exist, so they must be recorded in memory, not re-read at the end.
            epoch_values.setdefault(epoch, value)
            # Only the periodic (every-10-epochs) checkpoints feed the pruner, as before -- reporting a noisy
            # per-epoch value to a median pruner every epoch would prune good trials at random.
            if is_best or epoch in reported_epochs:
                continue
            reported_epochs.add(epoch)
            print(f"[Trial {trial.number}] epoch={epoch}, validation_significance={value:.5f}")
            trial.report(value, epoch)
            if trial.should_prune():
                print(f"[Trial {trial.number}] PRUNED at epoch {epoch} (validation_significance={value:.5f})")
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

    # Diagnostic extra, not part of the optimization itself -- runs only
    # for trials that completed normally (pruned trials exit early via
    # optuna.TrialPruned() raised from inside scan_for_new_checkpoints
    # above, so they never reach this point, and their partial data
    # isn't worth a plot anyway).
    loss_lr_plot_path = os.path.join(work_dir, f"{label}_loss_vs_lr.png")
    save_loss_vs_lr_plot(
        version_dir, loss_lr_plot_path, trial_number=trial.number,
        loss_tag_hint=getattr(args, "loss_tag", None),
        lr_tag_hint=getattr(args, "lr_tag", None),
    )

    score = trial_score(epoch_values)
    if score is not None:
        print(f"[Trial {trial.number}] score = mean of top {min(TOP_K_EPOCHS, len(epoch_values))} epochs "
              f"{sorted(epoch_values.values(), reverse=True)[:TOP_K_EPOCHS]} = {score:.5f}")
    return score if score is not None else 0.0


#ADJUST THESE -- this is now the ONLY place the active search space is defined.
# Both make_objective() below and main()'s GridSampler setup read from this one dict,
# so there's no second copy to keep in sync by hand (that's what let the dropout edit
# silently turn into a bare set literal instead of a dict entry -- a stray uncommented
# line with no matching "key: " couldn't happen here, since each entry is a real dict
# item with its name given once, as the key).
#
# kind is one of:
#   "categorical" -- spec is an explicit list of values (required for --sampler grid,
#                     which walks a fixed list -- a continuous range has no list to walk)
#   "float"       -- spec is (low, high), sampled uniformly
#   "float_log"   -- spec is (low, high), sampled log-uniformly
#   "int"         -- spec is (low, high) or (low, high, step)
#
# For a one-at-a-time diagnostic sweep (--sampler grid), exactly ONE entry below should
# be uncommented. --sampler tpe (the default) is fine with several uncommented at once --
# it searches them jointly.
ADJUST_PARAMS = {
    "learning_rate": ("float_log", (1e-4, 2e-3)),
    "dropout": ("float", (0.05, 0.30)),
    "l2_penalty": ("float_log", (1e-5, 1e-3)),
    "focal_gamma": ("float", (0.0, 1.5)),
    "batch_size": ("categorical", [256, 512, 1024]),
    "hidden_dim": ("categorical", [64, 96, 128]),
    "classification_loss_scale": ("float_log", (0.1, 3.0)),
}

# DON'T TOUCH THESE YET -- reference only, not read by anything. To scan one of these
# instead, move its line up into ADJUST_PARAMS above (same (kind, spec) shape).
_FIXED_PARAMS_REFERENCE = {
    #"hidden_dim": ("categorical", [64, 96, 128, 160, 192, 224, 256]),
    #"combine_pair_loss": ("categorical", ["min", "softmin", "mean"]),
    #"num_encoder_layers": ("int", (2, 10)),
    #"num_branch_embedding_layers": ("int", (1, 8)),
    #"num_branch_encoder_layers": ("int", (1, 8)),
    #"num_jet_embedding_layers": ("int", (0, 4)),
    #"num_jet_encoder_layers": ("int", (0, 4)),
    #"num_attention_heads": ("categorical", [1, 2, 4, 8]),
    #"assignment_loss_scale": ("float", (0.0, 1.0)),
    #"detection_loss_scale": ("float", (0.0, 1.0)),
}


def suggest_param(trial, name, spec):
    kind, values = spec
    if kind == "categorical":
        return trial.suggest_categorical(name, values)
    if kind == "float":
        low, high = values
        return trial.suggest_float(name, low, high)
    if kind == "float_log":
        low, high = values
        return trial.suggest_float(name, low, high, log=True)
    if kind == "int":
        if len(values) == 3:
            low, high, step = values
            return trial.suggest_int(name, low, high, step=step)
        low, high = values
        return trial.suggest_int(name, low, high)
    raise ValueError(f"Unknown suggestion kind {kind!r} for parameter {name!r} in ADJUST_PARAMS")


def make_objective(args, work_dir):
    def objective(trial):
        overrides = {name: suggest_param(trial, name, spec) for name, spec in ADJUST_PARAMS.items()}
        return run_trial_with_pruning(trial, overrides, args, work_dir)
    return objective


def save_param_importance_plot(study, output_path):
    """Bar chart of each hyperparameter's estimated importance, using
    optuna.importance.get_param_importances() directly -- NOT
    optuna.visualization.plot_param_importances(), which returns a
    Plotly figure. Saving THAT to a static image requires kaleido, and
    kaleido v1+ requires an actual Chrome installation to render at all
    -- confirmed directly to fail with ChromeNotFoundError in a headless
    environment. get_param_importances() returns a plain dict with no
    plotting library involved, so rendering it ourselves with
    matplotlib (already a proven dependency here) avoids that entirely.
    Requires at least a handful of completed trials to be meaningful --
    with very few trials this will run but the values won't mean much.
    """
    import optuna
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as e:
        print(f"Could not compute parameter importances (often just means too few completed "
              f"trials yet): {e}")
        return

    if not importances:
        print("No parameter importances to plot (no completed trials?)")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        names = list(importances.keys())
        values = list(importances.values())
        # Sort ascending so the most important parameter ends up at the
        # TOP of a horizontal bar chart, matching how these are
        # conventionally read.
        order = sorted(range(len(names)), key=lambda i: values[i])
        names = [names[i] for i in order]
        values = [values[i] for i in order]

        fig_height = max(0.35 * len(names) + 1.0, 2.5)
        fig, ax = plt.subplots(figsize=(8, fig_height))
        ax.barh(names, values, color="#2a78d6")
        ax.set_xlabel("Relative importance")
        ax.set_title("Optuna hyperparameter importance")
        for i, v in enumerate(values):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Saved parameter importance plot to: {output_path}")
    except ImportError:
        print("matplotlib not available -- skipped parameter importance plot")


def save_summary_outputs(study, output_prefix, seed=None):
    """Writes the scan summary to both a .txt file and a presentation-ready
    .png table, in addition to the normal console output. Hyperparameter
    columns are derived dynamically from whatever the trials actually
    scanned, rather than hardcoded -- stays correct if the search space
    changes later without needing this function updated too."""
    out_dir = os.path.dirname(output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    param_names = []
    for t in study.trials:
        for k in t.params.keys():
            if k not in param_names:
                param_names.append(k)

    def fmt_param(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    rows = []
    for t in study.trials:
        acc_str = f"{t.value:.5f}" if t.value is not None else "N/A"
        row = [str(t.number), t.state.name, acc_str] + [
            fmt_param(t.params.get(p, "")) for p in param_names
        ]
        rows.append(row)

    headers = ["Trial", "State", "Significance"] + param_names

    # --- .txt output ---
    col_widths = [max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) for i in range(len(headers))]
    lines = []
    lines.append("Optuna hyperparameter scan results")
    if seed is not None:
        lines.append(f"Seed (train.py -r/--random_seed, fixed across every trial): {seed}")
    lines.append("=" * (sum(col_widths) + 2 * len(col_widths)))
    lines.append("  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)))
    lines.append("-" * (sum(col_widths) + 2 * len(col_widths)))
    for r in rows:
        lines.append("  ".join(r[i].ljust(col_widths[i]) for i in range(len(r))))
    if study.best_trial is not None:
        lines.append("")
        lines.append(f"Best trial: #{study.best_trial.number}, significance={study.best_value:.5f}")
        lines.append(f"Best params: {study.best_params}")

    txt_path = f"{output_prefix}.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved summary text to: {txt_path}")

    # --- .png output ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_rows = len(rows)
        n_cols = len(headers)

        # Per-column content length (header text OR longest data value in
        # that column, whichever is longer) -- so a column like
        # 'num_branch_embedding_layers' correctly gets more width than
        # 'dropout', rather than every column getting the same fixed
        # slice regardless of how long its actual content is.
        col_content_lengths = []
        for col_idx, header in enumerate(headers):
            max_len = len(header)
            for row in rows:
                max_len = max(max_len, len(row[col_idx]))
            col_content_lengths.append(max_len)
        total_content_length = sum(col_content_lengths)
        col_widths = [length / total_content_length for length in col_content_lengths]

        # Total figure width scaled to the REAL total content length, not
        # a flat per-column count -- with many long-named columns (13+
        # hyperparameters), a flat multiplier stayed too narrow even
        # though individual columns were proportioned correctly, since
        # proportional widths alone can't fix a canvas that's too small
        # overall. 0.11in/char is empirical, tuned to the fontsize below;
        # floored at 10in so a small scan (few short columns) doesn't
        # render a tiny, cramped table either.
        fig_height = 0.4 * (n_rows + 1) + 0.6
        fig_width = max(total_content_length * 0.11, 10)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")

        table = ax.table(
            cellText=rows, colLabels=headers, colWidths=col_widths,
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        best_trial_number = str(study.best_trial.number) if study.best_trial is not None else None
        for (row_idx, col_idx), cell in table.get_celld().items():
            if row_idx == 0:
                cell.set_facecolor("#2a78d6")
                cell.set_text_props(color="white", weight="bold")
            elif best_trial_number is not None and rows[row_idx - 1][0] == best_trial_number:
                cell.set_facecolor("#e6f1fb")

        title = "Optuna hyperparameter scan results"
        if study.best_trial is not None:
            title += f"  (best: trial #{best_trial_number}, significance={study.best_value:.5f})"
        ax.set_title(title, fontsize=11, pad=12)

        png_path = f"{output_prefix}.png"
        plt.savefig(png_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved summary image to: {png_path}")
    except ImportError:
        print("matplotlib not available -- skipped .png output (saved .txt only)")


def recover_interrupted_trials(study, storage):
    """Resume support for --study_db runs. If an earlier launch died mid-trial (server lost,
    pod evicted, terminal killed), Optuna leaves that trial in state RUNNING forever. Mark each
    such trial FAIL with an 'interrupted' flag, and re-enqueue the same hyperparameters as a
    new trial so that configuration is retrained from scratch (train.py has no mid-trial
    resume here). The flag is what keeps interrupted trials from counting toward --n_trials."""
    running = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
    if not running:
        return 0
    study_id = storage.get_study_id_from_name(study.study_name)
    requeued = 0
    for t in list(running):
        params = dict(t.params)
        trial_id = storage.get_trial_id_from_study_id_trial_number(study_id, t.number)
        storage.set_trial_user_attr(trial_id, "interrupted", True)
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)
        if all(name in params for name in ADJUST_PARAMS):
            study.enqueue_trial(params)
            requeued += 1
            print(f"[resume] trial {t.number} was interrupted; re-enqueued its parameters: {params}")
        else:
            print(f"[resume] trial {t.number} was interrupted before all parameters were suggested; "
                  f"marked FAIL, not re-enqueued")
    return requeued


def trials_consumed(study):
    """Trials that count toward --n_trials: finished (COMPLETE/PRUNED), plus FAILs that were real
    failures (crash, divergence). Interrupted trials and still-waiting trials do not count."""
    n = 0
    for t in study.get_trials(deepcopy=False):
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED):
            n += 1
        elif t.state == optuna.trial.TrialState.FAIL and not t.user_attrs.get("interrupted"):
            n += 1
    return n


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
    parser.add_argument("--seed", type=int, default=None, help="Fixed across every trial and passed as train.py's -r/--random_seed, which drives BOTH "
                                                                 "the dataset split (options.dataset_randomization) and the model/training randomness "
                                                                 "(set_seed()) together -- train.py has no separate dataset-seed flag, one value does "
                                                                 "both. Using the same value for every trial is what makes trials comparable at all "
                                                                 "(a hyperparameter scan needs every trial evaluated on the identical split), unlike the "
                                                                 "separate model-training seed sweep which wants each run to draw its own. If left "
                                                                 "unset, a fresh seed is drawn from OS entropy for this scan and printed/logged so the "
                                                                 "exact scan can be reproduced later by passing that value here.")
    parser.add_argument("--train_module", default=None, help="Defaults to 'spanet.train'")
    parser.add_argument("--work_dir", default=None, help="Defaults to './optuna_scan_configs'")
    parser.add_argument("--n_trials", type=int, default=None, help="Defaults to 18 (17 adaptive + the enqueued baseline). With --study_db this is the TOTAL for the study, so relaunching after an interruption only runs what is left.")
    parser.add_argument("--sampler", choices=["tpe", "grid"], default=None,
                         help="'tpe' (default) is Optuna's adaptive sampler -- good for an actual optimization "
                              "search, but gives NO guarantee of even coverage across a categorical search space "
                              "(it can cluster trials on values it judges promising early and skip others "
                              "entirely). 'grid' uses optuna.samplers.GridSampler over whichever single "
                              "hyperparameter is active in the ADJUST_PARAMS dict near the top of this file "
                              "instead -- walks the values systematically, so every value gets covered. Use "
                              "'grid' for the one-at-a-time diagnostic sweeps (exactly one ADJUST_PARAMS entry "
                              "uncommented); for full grid coverage plus the enqueued baseline, set --n_trials "
                              "to (number of values) + 1.")
    parser.add_argument("--prune_warmup", type=int, default=None,
                         help="MedianPruner n_warmup_steps, in epochs (trial.report is called with the checkpoint's "
                              "epoch number, i.e. 9, 19, 29, ...). No trial is pruned before this epoch. Defaults to "
                              "25 -- the validation accuracy of a typical run is still climbing at epoch 9-19 and "
                              "plateaus around epoch 30-50, so pruning from the first checkpoint kills slow starters "
                              "(low learning rate, heavy regularization) for being slow, not for being worse.")
    parser.add_argument("--poll_interval", type=float, default=30.0, help="Seconds between checkpoint-directory polls while a trial is running")
    parser.add_argument("--startup_timeout", type=int, default=300, help="Seconds to wait for a trial's version directory to appear before giving up. Dataset loading + feature normalization can genuinely take a while on a cold start -- this is deliberately generous, and now that stdout/stderr are captured to a log file, a real crash gets detected and reported immediately regardless of this value, rather than waiting out the full timeout.")
    parser.add_argument("--study_db", default=None, help="If set, path to a SQLite file for persisting the study (allows resuming across sessions). Otherwise the study exists only in memory for this run.")
    parser.add_argument("--summary_output", default=None, help="Path prefix for saved summary files -- writes <prefix>.txt and <prefix>.png. Defaults to '<work_dir>/scan_summary' if not given.")
    parser.add_argument("--loss_tag", default=None, help="Exact TensorBoard scalar tag to use as 'loss' in the per-trial loss-vs-learning-rate diagnostic plot. Leave unset to auto-detect by name (case-insensitive substring match on 'loss', excluding anything with 'val'/'validation' in it) -- the tag actually picked is printed for every trial, so pass this explicitly if that guess is wrong for your setup.")
    parser.add_argument("--lr_tag", default=None, help="Exact TensorBoard scalar tag to use as 'learning rate' in the per-trial loss-vs-learning-rate diagnostic plot. Leave unset to auto-detect (matches 'lr' or 'learning_rate' in the tag name) -- printed for every trial, so pass this explicitly if the guess is wrong.")
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
        args.seed = generate_seed()
        print(f"Seed: {args.seed}  (auto-generated -- pass --seed {args.seed} to reproduce this exact scan)")
    else:
        print(f"Seed: {args.seed}  (explicitly set)")
    if args.train_module is None:
        args.train_module = "spanet.train"
    if args.work_dir is None:
        args.work_dir = "./optuna_scan_configs"
    if args.n_trials is None:
        args.n_trials = 18
    if args.sampler is None:
        args.sampler = "tpe"
    if args.prune_warmup is None:
        args.prune_warmup = 25
    if args.summary_output is None:
        args.summary_output = os.path.join(args.work_dir, "scan_summary")

    required_fields = ["baseline", "event_file", "training_file", "validation_file", "log_dir"]
    missing = [f for f in required_fields if getattr(args, f) is None]
    if missing:
        raise SystemExit(f"Missing required argument(s), not provided via CLI flag or --run_config: {missing}")

    # One timestamp for the whole run, computed up front and reused for
    # both the per-trial work directory below and the summary output at
    # the end -- so everything this scan produces is grouped under a
    # matching timestamp, and so it's computed once rather than twice
    # (previously main() computed a second, separately-timed timestamp
    # just before saving the summary, which could drift from this one by
    # however long the scan took to run).
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Nest every trial's saved config/log/diagnostic-plot files under a
    # per-run timestamped subdirectory of work_dir, rather than writing
    # trial_0.json / trial_0.log / trial_0_loss_vs_lr.png directly into
    # work_dir. Those filenames are only unique WITHIN one scan run --
    # work_dir defaults to the same fixed path every time, so without
    # this, launching a second scan would silently overwrite the
    # previous scan's trial configs, logs, and loss-vs-lr plots as soon
    # as trial numbers repeat (trial_0 clobbers trial_0, etc.).
    run_work_dir = os.path.join(args.work_dir, run_timestamp)
    os.makedirs(run_work_dir, exist_ok=True)
    print(f"Trial configs/logs/plots for this run: {run_work_dir}")

    if args.sampler == "grid":
        # Reads directly from ADJUST_PARAMS (defined near the top of this file, right
        # above make_objective()) -- there is no second copy of the value list to keep
        # in sync by hand anymore. --sampler grid only makes sense for exactly one
        # categorical hyperparameter at a time (GridSampler walks a fixed value list),
        # so this fails loudly with a clear message rather than a cryptic traceback if
        # ADJUST_PARAMS doesn't currently look like that -- e.g. zero or multiple
        # entries uncommented, or the active entry is a continuous "float"/"float_log"
        # range with no fixed list to walk.
        if len(ADJUST_PARAMS) != 1:
            raise SystemExit(
                f"--sampler grid needs exactly ONE active (uncommented) entry in ADJUST_PARAMS, "
                f"found {list(ADJUST_PARAMS.keys())} -- comment out all but the one hyperparameter "
                f"you're grid-scanning this run."
            )
        grid_param, (grid_kind, grid_values) = next(iter(ADJUST_PARAMS.items()))
        if grid_kind != "categorical":
            raise SystemExit(
                f"--sampler grid needs the active ADJUST_PARAMS entry ({grid_param!r}) to be "
                f"'categorical' (an explicit value list) -- got {grid_kind!r}, which has no fixed "
                f"list of values for GridSampler to walk."
            )
        sampler = optuna.samplers.GridSampler({grid_param: grid_values})
        full_coverage_trials = len(grid_values) + 1  # +1 for the enqueued baseline trial below
        print(f"Using GridSampler over {grid_param}={grid_values} -- every value gets exactly one "
              f"adaptively-scheduled trial (the enqueued baseline is separate and doesn't count against the "
              f"grid). --n_trials is currently {args.n_trials}; {full_coverage_trials} is needed for full "
              f"one-pass coverage of every value plus the baseline.")
        if args.n_trials < full_coverage_trials:
            print(f"WARNING: --n_trials={args.n_trials} is fewer than {full_coverage_trials} -- "
                  f"not every value in {grid_param}={grid_values} will get a trial this run.")
    else:
        sampler = optuna.samplers.TPESampler()

    storage = None
    if args.study_db:
        storage = optuna.storages.RDBStorage(f"sqlite:///{args.study_db}")
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=args.prune_warmup),
            storage=storage,
            study_name=args.name,
            load_if_exists=True,
        )
        recover_interrupted_trials(study, storage)
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=optuna.pruners.MedianPruner(n_warmup_steps=args.prune_warmup))

    # Enqueue the exact current baseline as an explicit trial, evaluated
    # under IDENTICAL conditions (same epoch budget, same seed) as every
    # adaptively-suggested trial -- not just a separately-run reference.
    with open(args.baseline) as f:
        baseline_values = json.load(f)
    baseline_defaults = {
        "hidden_dim": 128, "batch_size": 256, "dropout": 0.2, "l2_penalty": 0.0002,
        "focal_gamma": 1.0, "learning_rate": 0.001, "classification_loss_scale": 1.0,
    }
    # Only parameters that are actually part of the search space are enqueued -- the
    # baseline's other settings reach every trial through the merged baseline JSON
    # anyway, and an enqueued value for a parameter the objective never suggests is
    # at best ignored and at worst a warning.
    baseline_trial_params = {
        name: baseline_values.get(name, baseline_defaults.get(name))
        for name in ADJUST_PARAMS
    }
    for name, value in baseline_trial_params.items():
        kind, spec = ADJUST_PARAMS[name]
        if value is None:
            print(f"WARNING: baseline JSON has no value for {name!r} and there is no default -- "
                  f"trial 0 will sample it like any other trial.")
        elif kind == "categorical" and value not in spec:
            print(f"WARNING: baseline {name}={value} is not one of the categorical choices {spec} -- "
                  f"trial 0 will NOT be the exact baseline.")
        elif kind == "float_log" and value <= 0:
            # Optuna cannot represent a non-positive value in a log-scaled parameter at all
            # (ValueError at suggest time, which kills trial 0), so this one is dropped from
            # the enqueued baseline and TPE samples it like any other trial.
            print(f"WARNING: baseline {name}={value} is not valid for a log-scaled range {spec[:2]} -- "
                  f"dropping it from the enqueued baseline; trial 0 will sample {name!r} instead of "
                  f"using the baseline value. Fix the baseline JSON to get an exact baseline trial.")
            baseline_trial_params[name] = None
        elif kind != "categorical" and not (spec[0] <= value <= spec[1]):
            print(f"WARNING: baseline {name}={value} is outside the search range {spec[:2]} -- "
                  f"trial 0 will run it as given, but TPE has no samples out there.")
    study.enqueue_trial({k: v for k, v in baseline_trial_params.items() if v is not None},
                        skip_if_exists=True)

    objective = make_objective(args, run_work_dir)
    # catch=(RuntimeError,) is critical here: without it, ANY RuntimeError
    # raised from a single trial (e.g. train.py crashing, or SPANet's own
    # "Assignment loss has diverged!" guard firing on an unstable
    # hyperparameter combination) stops the ENTIRE study immediately,
    # losing every trial that would have come after it -- confirmed
    # directly against Optuna's actual default behavior, not assumed.
    # optuna.TrialPruned (used for pruning) is handled separately by
    # Optuna itself regardless of this setting.
    if args.study_db:
        # --n_trials is the TOTAL for the study, so relaunching the same command after an
        # interruption finishes the remaining trials instead of running n_trials more.
        done = trials_consumed(study)
        n_to_run = max(0, args.n_trials - done)
        print(f"[resume] study '{args.name}' in {args.study_db}: {done} trial(s) already counted, "
              f"{n_to_run} of {args.n_trials} left to run")
    else:
        n_to_run = args.n_trials
    if n_to_run > 0:
        study.optimize(objective, n_trials=n_to_run, catch=(RuntimeError,))

    print("\n" + "=" * 90)
    print("SUMMARY: Optuna hyperparameter scan results (objective: validation_significance)")
    print("=" * 90)
    print(f"{'Trial':<8} {'State':<10} {'Significance':>12}  Params")
    print("-" * 90)
    for t in study.trials:
        acc_str = f"{t.value:.5f}" if t.value is not None else "N/A"
        print(f"{t.number:<8} {t.state.name:<10} {acc_str:>12}  {t.params}")
    print("-" * 90)
    if study.best_trial is not None:
        print(f"Best trial: #{study.best_trial.number}, significance={study.best_value:.5f}")
        print(f"Best params: {study.best_params}")
    print("=" * 90)

    # Timestamp appended regardless of whether summary_output came from the
    # default or was explicitly given -- so repeated runs never silently
    # overwrite a previous run's saved summary, even if --summary_output
    # points at the same directory every time. Reuses the same
    # run_timestamp computed at the top of main(), so this matches the
    # trial work directory's own timestamp for this run.
    timestamped_summary_output = f"{args.summary_output}_{run_timestamp}"
    save_summary_outputs(study, timestamped_summary_output, seed=args.seed)
    save_param_importance_plot(study, f"{timestamped_summary_output}_importance.png")

    return study


if __name__ == "__main__":
    main()