#!/usr/bin/env bash
# Copy the Pi's data down to the Mac for labeling and training.
#
# Read-only on the Pi: rsync pulls, nothing is pushed, and nothing under
# ~/dog-sleep-monitor is touched. The live monitor keeps running throughout.
#
# Everything lands in .local/pi/, which is gitignored. That matters: the
# archive is pictures of the inside of a home and this repo is public.
#
#   ./train/pull.sh                 # default host
#   PI=josue@10.0.0.5 ./train/pull.sh
set -euo pipefail

PI="${PI:-josue@100.102.155.66}"
SRC="${SRC:-dog-sleep-monitor}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HERE/.local/pi"

mkdir -p "$DEST"

echo "Pulling from $PI:~/$SRC/ into ${DEST#"$HERE"/}/"

# The archive is the bulk of it: ~39 MB per hour at every-sample archiving.
# --partial so a dropped Wi-Fi connection resumes instead of restarting, and
# no --delete: frames the Pi's ring buffer has already dropped are still worth
# keeping here, since a label window may point at them.
rsync -av --partial --info=progress2 \
    "$PI:~/$SRC/archive/" "$DEST/archive/"

# The numbers. Small, and the ones that give the frames their context.
for f in sleep_log.csv events.csv presence_labels.csv markers.csv config.json; do
    rsync -av "$PI:~/$SRC/$f" "$DEST/$f" 2>/dev/null \
        || echo "  (no $f on the Pi yet, skipping)"
done

echo
echo "Frames:  $(find "$DEST/archive" -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')"
echo "Size:    $(du -sh "$DEST" | cut -f1)"
echo
echo "Next: label windows, then export a manifest."
echo "  python monitor.py label-presence --from ... --to ... --label dog"
echo "  python monitor.py dataset --archive .local/pi/archive --labeled-only"
