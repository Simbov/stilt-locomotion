---
name: repo-privacy
description: Repo is PUBLIC on GitHub; keep personal identifiers out; local-only files that must never be committed
metadata: 
  node_type: memory
  type: project
  originSessionId: f6743e3d-2c3e-4cb9-9b53-3f1d4d00e86a
  modified: 2026-07-23T03:02:09.421Z
---

The parent repo is **public** at `github.com/Simbov/stilt-locomotion` (remote
`git@github.com:Simbov/stilt-locomotion.git`). User wants nothing private in it —
only outsider-relevant project content.

**Never commit (gitignored, kept local-only on disk):**
- `scripts/sync_logs.sh` — holds the personal HPC login (`n11298111@aquarius02.hpc.qut.edu.au`).
- `.claude/settings.local.json` — machine-specific paths + local Claude permissions.

**Genericize in any committed doc:** HPC username → `<hpc-user>`, W&B entity
`simbov04-qut` → `<wandb-entity>`. The QUT hostname `aquarius02.hpc.qut.edu.au`
is a public resource and may stay. Robot factory creds (`unitree`/`123`,
`192.168.123.164`) in `deploy/README.md` are the documented Unitree default —
user chose to KEEP them.

**Constraint (2026-07-23):** user does NOT want history rewrites / force-pushes —
they must not break the local↔remote sync workflow. So privacy fixes are
"going forward only." Consequence: old public commits (e.g. `d2cd4eb`) still
contain the HPC username; considered already-exposed, not worth a rewrite.

**Why:** avoid leaking the user's QUT HPC login and personal machine details on a
public repo. **How to apply:** before committing/pushing, grep staged content for
`n11298111|simbov04|/Users/simonvollert`; keep the two files above out of git.
