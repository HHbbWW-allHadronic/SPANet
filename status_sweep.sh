#!/usr/bin/env bash
#
# Status/progress table for a seed-sweep launched by launch_seed_sweep.sh.
# Reads the manifest (job/gpu/pid/logfile), and for each job reports:
#   - alive/stopped, via the recorded PID
#   - most recent "Epoch X/Y" line seen in its log (live progress -- Lightning's
#     progress bar does write plain "Epoch X/Y" lines even to a redirected,
#     non-tty log file, so this is a real-time signal, not just a guess)
#   - for each binning method (score bins, signal-efficiency bins, mass bins):
#       * LAST: the value on the most recent Lightning line "'<metric>' reached X" -- Lightning only prints that line when
#         the metric improves, so LAST is the most recent IMPROVEMENT, not necessarily the latest epoch's value
#       * BEST: the best value so far, read off the saved checkpoint filenames. train.py writes the best-*, bestEff-* and
#         bestMass-* checkpoints EVERY epoch (top 3), so this is current to within one epoch.
#
# Usage: ./status_sweep.sh <manifest_file> [--html <output.html>] [--refresh <seconds>]
#   e.g.: ./status_sweep.sh seed_sweep_logs/seedsweep_20260922_manifest.tsv
#   e.g.: ./status_sweep.sh seed_sweep_logs/seedsweep_20260922_manifest.tsv --html seed_sweep_logs/dashboard.html
#
# Re-run any time -- it's read-only, safe to run while jobs are still going.
# For a live terminal view with no code changes at all: `watch -n 10 ./status_sweep.sh <manifest>`
# For a live HTML view that auto-refreshes: use watch_dashboard.sh, which calls this on a loop.

set -uo pipefail

MANIFEST=""
HTML_PATH=""
REFRESH_SECONDS=10
# One column pair (last / best) per binning method; must match launch_seed_sweep.sh and the monitors in train.py.
METRICS=(validation_significance validation_significance_eff validation_significance_mass)
LABELS=(SCORE EFF MASS)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --html)
            HTML_PATH="$2"; shift 2 ;;
        --refresh)
            REFRESH_SECONDS="$2"; shift 2 ;;
        *)
            if [[ -z "$MANIFEST" ]]; then MANIFEST="$1"; shift; else echo "Unexpected argument: $1" >&2; exit 1; fi
            ;;
    esac
done

if [[ -z "$MANIFEST" ]]; then
    echo "Usage: $0 <manifest_file> [--html <output.html>] [--refresh <seconds>]"
    exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST"
    exit 1
fi

TRAIN_LOG_DIR=$(grep -m1 '^# TRAIN_LOG_DIR=' "$MANIFEST" | cut -d= -f2-)
if [[ -z "$TRAIN_LOG_DIR" ]]; then
    echo "Warning: couldn't find '# TRAIN_LOG_DIR=' header in $MANIFEST -- defaulting to '.'"
    TRAIN_LOG_DIR="."
fi

ROW_FMT="%-24s %-4s %-8s %-9s %-14s"
HDR1=(JOB GPU PID STATUS EPOCH)
HDR2=(--- --- --- ------ -----)
for l in "${LABELS[@]}"; do
    ROW_FMT+=" %-9s %-9s"
    HDR1+=("$l last" "$l best")
    HDR2+=("--------" "--------")
done
printf "$ROW_FMT\n" "${HDR1[@]}"
printf "$ROW_FMT\n" "${HDR2[@]}"

if [[ -n "$HTML_PATH" ]]; then
    HTML_TMP="${HTML_PATH}.tmp.$$"
    {
        echo "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        echo "<meta http-equiv='refresh' content='$REFRESH_SECONDS'>"
        echo "<title>Seed sweep status</title>"
        cat <<'CSS'
<style>
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; background:#0f1115; color:#e6e6e6; padding:24px; }
h1 { font-size:1.2rem; font-weight:600; margin-bottom:4px; }
.meta { color:#9aa0a6; font-size:0.85rem; margin-bottom:16px; }
table { border-collapse: collapse; width:100%; }
th, td { text-align:left; padding:8px 12px; border-bottom:1px solid #2a2d34; font-size:0.9rem; }
th { color:#9aa0a6; font-weight:600; text-transform:uppercase; font-size:0.75rem; letter-spacing:0.03em; }
.status-running { color:#3ddc84; font-weight:600; }
.status-stopped { color:#9aa0a6; }
tr:hover td { background:#1a1d24; }
</style>
CSS
        echo "</head><body>"
        echo "<h1>Seed sweep: $(basename "$MANIFEST")</h1>"
        echo "<div class='meta'>Last updated: $(date '+%Y-%m-%d %H:%M:%S %Z') &middot; auto-refreshes every ${REFRESH_SECONDS}s</div>"
        echo "<table><thead><tr><th>Job</th><th>GPU</th><th>PID</th><th>Status</th><th>Last Epoch</th>$(for l in "${LABELS[@]}"; do echo -n "<th>$l last</th><th>$l best</th>"; done)</tr></thead><tbody>"
    } > "$HTML_TMP"
fi

# skip the '# TRAIN_LOG_DIR=' comment line and the TSV header line
tail -n +3 "$MANIFEST" | while IFS=$'\t' read -r job_idx name gpu pid logfile start_time; do
    [[ -z "${name:-}" ]] && continue

    if kill -0 "$pid" 2>/dev/null; then
        status="running"
    else
        status="stopped"
    fi

    last_epoch=$(grep -oE "Epoch [0-9]+/[0-9]+" "$logfile" 2>/dev/null | tail -n 1)
    [[ -z "$last_epoch" ]] && last_epoch="n/a"

    # TensorBoardLogger doesn't overwrite an existing version_0 for a reused
    # --name -- it creates version_1, version_2, etc. Always resolve the
    # latest one rather than assuming version_0, so a reused run_label (e.g.
    # a second test run) doesn't silently read a stale, older run's checkpoints.
    latest_version_dir=$(ls -d "$TRAIN_LOG_DIR/$name"/version_* 2>/dev/null | sort -V | tail -n 1)
    ckpt_dir="${latest_version_dir:-}/checkpoints"

    vals=()
    for metric in "${METRICS[@]}"; do
        last_sig=$(grep -oE "'${metric}' reached [0-9]+\.[0-9]+" "$logfile" 2>/dev/null | tail -n 1 | awk '{print $NF}')
        [[ -z "$last_sig" ]] && last_sig="n/a"

        best_acc="n/a"
        if [[ -d "$ckpt_dir" ]]; then
            # "${metric}=" with the "=" so validation_significance does not also match the _eff / _mass checkpoints
            parsed=$(ls "$ckpt_dir" 2>/dev/null \
                | grep -oE "${metric}=[0-9]+\.[0-9]+" \
                | cut -d= -f2 \
                | sort -g \
                | tail -n 1)
            [[ -n "$parsed" ]] && best_acc="$parsed"
        fi
        vals+=("$last_sig" "$best_acc")
    done

    printf "$ROW_FMT\n" "$name" "$gpu" "$pid" "$status" "$last_epoch" "${vals[@]}"

    if [[ -n "$HTML_PATH" ]]; then
        status_class="status-stopped"
        [[ "$status" == "running" ]] && status_class="status-running"
        cells=""
        for v in "${vals[@]}"; do cells+="<td>$v</td>"; done
        echo "<tr><td>$name</td><td>$gpu</td><td>$pid</td><td class='$status_class'>$status</td><td>$last_epoch</td>$cells</tr>" >> "$HTML_TMP"
    fi
done

if [[ -n "$HTML_PATH" ]]; then
    echo "</tbody></table></body></html>" >> "$HTML_TMP"
    mv "$HTML_TMP" "$HTML_PATH"
fi