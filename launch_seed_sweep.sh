#!/usr/bin/env bash
#
# Launch N SPANet training runs (the model-training seed sweep -- same
# hyperparameters/options file, but BOTH --dataset_seed and --random_seed
# left unset so train.py auto-generates and logs its own pair per run --
# "set everything or set nothing", and this stage sets nothing) spread
# across a set of GPUs, running in batches of len(GPUS) jobs at a time:
# fill every GPU, wait for that whole batch to finish, then launch the next.
#
# Each job is launched with nohup so it survives a hangup signal on its own.
# To make the WHOLE run (launcher + all its batches) survive your terminal
# closing or your connection dropping, run this script itself under nohup --
# see the usage note below. (Don't add `disown` on the individual per-job
# PIDs inside the batch loop -- that breaks this script's own `wait` for
# those PIDs, which is what makes "wait for the batch to finish" work at
# all. nohup alone is enough to protect each job from SIGHUP.)
#
# Usage:
#   ./launch_seed_sweep.sh <total_jobs> <run_label>          # foreground
#   nohup ./launch_seed_sweep.sh <total_jobs> <run_label> \  # survives terminal closing
#       > launcher.log 2>&1 &
#
# Example: 18 jobs, 6 GPUs -> 3 batches of 6.
#   nohup ./launch_seed_sweep.sh 18 seedsweep_20260922 > launcher.log 2>&1 &
#
# Produces (all nested under spanet_output/ to keep the repo root clean):
#   spanet_output/seed_sweep_logs/<run_label>_manifest.tsv       -- job/gpu/pid/logfile table, read by status_sweep.sh
#   spanet_output/seed_sweep_logs/<run_label>_job000.log ...      -- one stdout/stderr log per job
#   spanet_output/<run_label>/<job_name>/version_N/...            -- TensorBoardLogger's checkpoints + options.json per run,
#                                                                      grouped under a subdirectory per run_label

set -uo pipefail

# Always run from this script's own directory (the SPANet repo root), regardless
# of where it was invoked from -- OPTIONS_FILE below and `python -m spanet.train`
# itself both depend on cwd being the repo root, and a relative path like
# "options_files/..." would silently resolve against the wrong directory otherwise.
cd "$(dirname "${BASH_SOURCE[0]}")"

# ---------------------------------------------------------------------------
# Config -- edit these for your setup
# ---------------------------------------------------------------------------
GPUS=()                         # leave empty to auto-detect via nvidia-smi (all GPUs
                                 # visible to this allocation); or override with specific
                                 # indices to exclude one, e.g. GPUS=(0 1 2 4 5)
OPTIONS_FILE="options_files/full_hadronic_HHbbWW/HHbbVV_training.json"
LOG_DIR="spanet_output/seed_sweep_logs"   # per-job stdout/stderr + manifest go here
TRAIN_LOG_DIR="spanet_output"             # passed to train.py's --log_dir (TensorBoardLogger root)
EXTRA_ARGS=(--verbose)          # any other fixed flags for every job
# Checkpoint-filename metrics used to pick the overall best model, one report per binning method. They must match the
# monitor= of the ModelCheckpoints in train.py (score bins / equal-signal-efficiency bins / mass grid x score bins).
# (Old sweeps: use METRICS=(validation_average_jet_accuracy) LABELS=(accuracy).)
METRICS=(validation_significance validation_significance_eff validation_significance_mass)
LABELS=("score bins" "signal-efficiency bins" "mass bins")
STAGGER_SECONDS=2               # gap between launches within a batch
# ---------------------------------------------------------------------------

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <total_jobs> <run_label>"
    echo "  e.g.: $0 18 seedsweep_20260922"
    exit 1
fi

TOTAL_JOBS=$1
RUN_LABEL=$2

if [[ ${#GPUS[@]} -eq 0 ]]; then
    if ! command -v nvidia-smi &> /dev/null; then
        echo "GPUS not set and nvidia-smi isn't available -- can't auto-detect. Set GPUS=(...) manually at the top of this script." >&2
        exit 1
    fi
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    if [[ ${#GPUS[@]} -eq 0 ]]; then
        echo "nvidia-smi reported zero GPUs -- nothing to launch on." >&2
        exit 1
    fi
    echo "Auto-detected GPUs from nvidia-smi: ${GPUS[*]}"
fi

NUM_GPUS=${#GPUS[@]}

# Nest every job in this run under its own run_label subdirectory
# (spanet_output/<run_label>/<run_label>_job000/version_N/...) instead of
# dumping all runs flat into spanet_output/ -- keeps the top level tidy as
# more sweeps pile up. status_sweep.sh needs no changes for this: it just
# reads whatever path is recorded in the manifest's TRAIN_LOG_DIR header.
RUN_LOG_DIR="$TRAIN_LOG_DIR/$RUN_LABEL"

mkdir -p "$LOG_DIR" "$RUN_LOG_DIR"
MANIFEST="$LOG_DIR/${RUN_LABEL}_manifest.tsv"

# First line records the log_dir used, so status_sweep.sh doesn't need a
# separately hand-synced copy of this config.
echo "# TRAIN_LOG_DIR=$RUN_LOG_DIR" > "$MANIFEST"
echo -e "job_idx\tname\tgpu\tpid\tlogfile\tstart_time" >> "$MANIFEST"

echo "Launching $TOTAL_JOBS jobs across ${NUM_GPUS} GPUs (${GPUS[*]}), in batches of ${NUM_GPUS}."
echo "Options file: $OPTIONS_FILE | dataset_seed and random_seed: unset, each job draws and logs its own"
echo "Manifest: $MANIFEST"
echo

job_idx=0
while [[ $job_idx -lt $TOTAL_JOBS ]]; do
    batch_pids=()
    batch_start=$job_idx
    batch_end=$(( job_idx + NUM_GPUS - 1 ))
    [[ $batch_end -ge $TOTAL_JOBS ]] && batch_end=$(( TOTAL_JOBS - 1 ))

    echo "=== Batch: jobs $batch_start-$batch_end ==="

    for (( i=batch_start; i<=batch_end; i++ )); do
        gpu_slot=$(( i - batch_start ))
        gpu=${GPUS[$gpu_slot]}
        job_name="${RUN_LABEL}_job$(printf '%03d' "$i")"
        logfile="$LOG_DIR/${job_name}.log"

        echo "  job $i -> GPU $gpu -> name=$job_name (log: $logfile)"

        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 nohup python -m spanet.train \
            -of "$OPTIONS_FILE" \
            --log_dir "$RUN_LOG_DIR" \
            --name "$job_name" \
            "${EXTRA_ARGS[@]}" \
            > "$logfile" 2>&1 &

        pid=$!
        batch_pids+=("$pid")
        echo -e "${i}\t${job_name}\t${gpu}\t${pid}\t${logfile}\t$(date -Iseconds)" >> "$MANIFEST"

        sleep "$STAGGER_SECONDS"
    done

    echo "Waiting for batch to finish (PIDs: ${batch_pids[*]}) ..."
    wait "${batch_pids[@]}"
    echo "Batch done."
    echo

    job_idx=$(( batch_end + 1 ))
done

echo "All $TOTAL_JOBS jobs complete."
echo "Manifest: $MANIFEST -- run ./status_sweep.sh $MANIFEST any time for a progress table."
echo

# ---------------------------------------------------------------------------
# Find the single best checkpoint across every job in this sweep -- not just
# each job's own best, the overall winner -- for EACH validation metric (one per
# binning method), plus the model/training seed that produced it (read back from
# that run's saved options.json), so it is fully reproducible. The three winners
# can be different jobs and different epochs: that is the comparison.
# A checkpoint filename carries its metric as "<metric>=<value>", e.g.
#   best-epoch=12-step=4784-validation_significance=0.01389.ckpt
#   bestEff-epoch=19-step=7360-validation_significance_eff=0.01412.ckpt
# so the "=" right after the metric name keeps validation_significance from matching the _eff / _mass files.
# ---------------------------------------------------------------------------
report_best() {
    local metric=$1 label=$2
    local best_val="" best_job="" best_gpu="" best_ckpt="" best_version_dir="" best_epoch=""
    local job_idx name gpu pid logfile start_time version_dir ckpt_dir ckpt val

    while IFS=$'\t' read -r job_idx name gpu pid logfile start_time; do
        [[ -z "${name:-}" ]] && continue

        # TensorBoardLogger doesn't overwrite an existing version_0 for a reused
        # --name -- it creates version_1, version_2, etc. Always resolve the
        # latest one rather than assuming version_0, so a reused run_label
        # doesn't silently pull in a stale, older run's checkpoints.
        version_dir=$(ls -d "$RUN_LOG_DIR/$name"/version_* 2>/dev/null | sort -V | tail -n 1)
        [[ -z "$version_dir" ]] && continue
        ckpt_dir="$version_dir/checkpoints"
        [[ -d "$ckpt_dir" ]] || continue

        while IFS= read -r ckpt; do
            [[ -z "$ckpt" ]] && continue
            val=$(echo "$ckpt" | grep -oE "${metric}=[0-9]+\.[0-9]+" | cut -d= -f2)
            [[ -z "$val" ]] && continue

            if [[ -z "$best_val" ]] || awk -v a="$val" -v b="$best_val" 'BEGIN{exit !(a>b)}'; then
                best_val="$val"
                best_job="$name"
                best_gpu="$gpu"
                best_ckpt="$ckpt_dir/$ckpt"
                best_version_dir="$version_dir"
                best_epoch=$(echo "$ckpt" | grep -oE "epoch=[0-9]+" | head -n 1 | cut -d= -f2)
            fi
        done < <(ls "$ckpt_dir" 2>/dev/null)
    done < <(tail -n +3 "$MANIFEST")

    echo "=== Best model across sweep by $label ($metric) ==="
    if [[ -z "$best_val" ]]; then
        echo "No checkpoints with $metric in the filename found across any job in $MANIFEST."
        echo
        return
    fi

    local random_seed="n/a" dataset_seed="n/a"
    local options_json="$best_version_dir/options.json"
    if [[ -f "$options_json" ]]; then
        read -r random_seed dataset_seed < <(python3 -c "
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get('random_seed', 'n/a'), d.get('dataset_randomization_seed', 'n/a'))
except Exception:
    print('n/a', 'n/a')
" "$options_json" 2>/dev/null)
        [[ -z "$random_seed" ]] && random_seed="n/a"
        [[ -z "$dataset_seed" ]] && dataset_seed="n/a"
    fi

    echo "Job:              $best_job   (epoch ${best_epoch:-n/a})"
    echo "GPU (at launch):  $best_gpu"
    echo "Best $metric: $best_val"
    echo "Random seed:      $random_seed"
    echo "Dataset seed:     $dataset_seed  (pass -r/--random_seed $random_seed --dataset_seed $dataset_seed to reproduce this exact run)"
    echo "Checkpoint file:  $best_ckpt"
    echo
}

for m in "${!METRICS[@]}"; do
    report_best "${METRICS[$m]}" "${LABELS[$m]}"
done