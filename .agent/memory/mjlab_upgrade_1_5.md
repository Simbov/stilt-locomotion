---
name: mjlab-upgrade-1-3-to-1-5
description: What changed and what to watch when upgrading the stilt env from mjlab 1.3 to 1.5
metadata: 
  node_type: memory
  type: project
  originSessionId: baad0fdb-702c-4687-95a0-4b91f2123849
  modified: 2026-07-23T02:45:39.907Z
---

Upgraded mjlab 1.3.0 → 1.5.0 on 2026-06-30. Bumped the PyPI pin in `pyproject.toml`
(`mjlab==1.5.0`), dropped the project's `mujoco<3.8` / `mujoco-warp==3.7.0.1` pins
(let mjlab drive them), checked out submodule to `v1.5.0`, ran `uv sync`.

**2026-07-23 patch bump 1.5.0 → 1.5.3** (start of semester 2). Same procedure:
`git -C mjlab checkout v1.5.3`, pin → `mjlab==1.5.3`, `uv sync`. mujoco-warp went
3.10.0.1 → **3.10.0.3**; mujoco stayed **3.10.0** (pin still `~=3.10.0`, so the 3.8
leak stays bypassed — nothing reopened). Relevant fixes in the range: NaN in
`bad_orientation` termination (unclamped acos), exact substep-dt for contact
air-time, raycast-sensor cache invalidation (that's `foot_height_scan`), richer
ONNX export metadata + W&B ONNX-upload fix (helps `deploy/`). Drop-in — no API the
stilt env overrides was removed. CPU 1-env smoke test (build→reset→3 steps,
`stilt_mass` DR event present) passed.

**Caveat:** the 1.3→1.5.0 upgrade was never committed — it lived only in the
working tree (parent HEAD still recorded submodule at v1.3.0 / `mjlab==1.3.0`). The
1.5.3 bump supersedes that dangling state; commit it as one clean 1.3→1.5.3 jump.

Resulting versions: mujoco **3.10.0**, mujoco-warp **3.10.0.3**, rsl-rl-lib **5.4.0**,
warp-lang **1.14.0**.

Breaking changes across 1.3→1.4→1.5 and how they hit this project:
- **mujoco/warp 3.7 → 3.10** (1.4 went to 3.8, 1.5 pinned 3.10 from PyPI; the old
  `py.mujoco.org` nightly index + mujoco-warp git pin are gone). The project's
  `mujoco<3.8` pin *conflicts* and had to be removed.
- **`multiccd` enableflag removed** (1.4, now always on) — NOT used in project code.
- **`ls_parallel` deprecated/ignored** (1.5) — NOT used in project code.
- **Stricter shape validation** in reward/termination/metrics managers (1.4) — the
  stilt env config builds and the full sim runs clean, so the custom stilt-mass
  curriculum + reward overrides survived it.

Verified: `import envs.stilt_g1` (config build + `register_mjlab_task`) works, and a
full 1-env CPU sim (build → reset → 3 steps → `stilt_mass` DR event present) passes.

**Open watch-item:** mujoco was pinned `<3.8` originally to dodge a 3.8.0 ~670 MB/iter
memory leak. 1.5 bypasses 3.8 to 3.10 — leak status on 3.10 is UNVERIFIED. Check GPU
memory over a long HPC run before trusting it. See [[stilt_mass_dr_investigation]].

**Why:** keep the project on the latest mjlab so future upgrades stay small.
**How to apply:** for the next upgrade, `cd mjlab && git checkout <tag>`, bump the
PyPI pin, `uv sync`, then re-run the import + full-sim smoke test above. Supersedes
the older [[mjlab_api_changes]] (1.2→1.3) notes.
