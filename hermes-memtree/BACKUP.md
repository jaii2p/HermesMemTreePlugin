# Backups — so a dead box never costs a memory

The trees are the real thing: the local one here (`var/`), and any second tree you
point at with `MEMTREE_PEER_HOME`. If a VPS or this box dies, a memory is a memory
lost. Back them up.

## Do it now
    ./backup.sh              # snapshots every tree found into backups/
    ./backup.sh list         # what's saved
    ./backup.sh verify local # tarball opens, memories intact, sha matches
    ./backup.sh verify peer

Verified output from a real run:
    local -> 3 memories restorable,   sha f1ec33bcd3f5…
    peer  -> 400 memories restorable, sha 750cc593309a…

## Automate it
    crontab backup.cron      # nightly 03:17 (installs/replaces your crontab — check it)
or add the line from backup.cron to `crontab -e`.

## Restore (only if something actually died)
    ./backup.sh restore local      # asks first; keeps the current tree aside
    ./backup.sh restore peer

## Make it a real plan
- Keep the last 20 per tree by default (`KEEP=20 ./backup.sh`), or set `KEEP`.
- **Offsite:** copy `backups/` somewhere else — rsync to a second box, an S3 bucket,
  a spare disk. `BACKUP_ROOT=/mnt/spare ./backup.sh`.
- A backup that only lives on the same machine it's protecting isn't a backup.

## Notes
- A backup includes the memory tree (`tree.json`), the journal, and the watchdog's
  mute + timer state — the whole inner life.
- `restore` never throws the current tree away: it moves it aside first.
- Backups are plain `.tar.gz` plus a `SHA256SUMS`/`manifest.json` per label,
  inspectable by hand.
