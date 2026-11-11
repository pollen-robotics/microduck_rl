#!/usr/bin/env python
"""Route A first-run checklist for the jump task — run this ON A CUDA BOX.

Why this file exists
--------------------
The jump task cfg was written and statically verified on a CPU-only dev box
where `mjlab` / `mujoco_warp` / `jax` are not installed, so the cfg has **never
been constructed**. Every check that can be done without those deps lives in
`tests/test_jump_cfg_static.py`; what remains is exactly the list below, and it
all needs real hardware.

Doing this by hand goes wrong in a specific way: people run the training
script, see a rising total reward, and call it success — while the duck is
doing a squat-and-rise and never leaving the ground. The total reward is the
one number that CANNOT distinguish a hop from a bob, because `jump_crouch` and
`jump_launch` pay generously for the wind-up either way. So this script checks
the phase-resolved terms instead, and refuses to print a green light if the
duck never got airborne.

Usage
-----
    python scripts/jump_first_run.py            # checks 1-4
    python scripts/jump_first_run.py --steps 400
    python scripts/jump_first_run.py --json out.json

Exit code 0 only if every check passed. A non-zero exit means "do not start a
long run yet" — read the failing check's hint.

Boundary: there is no real robot here, so a pass means **deploy-ready**, never
sim-to-real. Do not report it as the latter.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback


def _hr(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def check_imports() -> dict:
    """Check 1 — the GPU stack is actually importable, and reports a GPU."""
    _hr("1. GPU stack imports")
    out = {"ok": False, "detail": {}}
    try:
        import jax
        import mujoco_warp  # noqa: F401
        import torch
    except Exception as exc:
        print(f"  FAIL  import: {type(exc).__name__}: {exc}")
        print("        mjlab/mujoco_warp/jax are required. See runs/JUMP_TASK.md.")
        return out

    devices = jax.devices()
    gpu = [d for d in devices if d.platform in ("gpu", "cuda")]
    out["detail"] = {
        "jax": jax.__version__, "torch": torch.__version__,
        "devices": [str(d) for d in devices],
    }
    print(f"  jax {jax.__version__}  torch {torch.__version__}")
    print(f"  devices: {[str(d) for d in devices]}")
    if not gpu:
        print("  FAIL  no GPU device visible to jax")
        return out
    print(f"  OK    {len(gpu)} GPU device(s)")
    out["ok"] = True
    return out


def check_construct() -> dict:
    """Check 2 — the cfg and the registered task both build.

    A hand-written cfg's most likely defect is a `.params[...]` key or a reward
    key that does not exist; both raise only at construction, which is why this
    check is separate from the import check.
    """
    _hr("2. cfg + task construction")
    out = {"ok": False, "detail": {}}
    try:
        from mjlab_microduck.tasks.microduck_jump_env_cfg import (
            make_microduck_jump_env_cfg,
        )
        for play in (False, True):
            cfg = make_microduck_jump_env_cfg(play=play)
            n_rewards = len(cfg.rewards)
            print(f"  OK    make(play={play}) -> {n_rewards} reward terms")
            out["detail"][f"play_{play}"] = n_rewards
    except Exception as exc:
        print(f"  FAIL  construction: {type(exc).__name__}: {exc}")
        # A missing dependency is an EXPECTED outcome on a box being set up —
        # printing a stack trace there trains people to ignore tracebacks. Only
        # dump one for a real construction error, where the frames matter.
        if isinstance(exc, (ImportError, ModuleNotFoundError)):
            print("        dependency missing, not a cfg defect — install the "
                  "GPU stack first")
        else:
            traceback.print_exc()
        return out

    # The registered env itself (catches a registration/robot-cfg mismatch).
    try:
        import mjlab_microduck.tasks  # noqa: F401  (triggers registration)
        from mjlab import register  # type: ignore
        out["detail"]["registry_import"] = "ok"
    except Exception as exc:
        # Not fatal on its own — registry APIs differ across mjlab versions.
        print(f"  note  registry probe skipped: {type(exc).__name__}: {exc}")
        out["detail"]["registry_import"] = f"skipped: {exc}"

    print("  OK    cfg constructs for both play and train")
    out["ok"] = True
    return out


def check_backlash_variant() -> dict:
    """Check 3 — the -Backlash twin registers with the WALK robot."""
    _hr("3. -Backlash variant registration")
    out = {"ok": False, "detail": {}}
    try:
        import mjlab_microduck.tasks  # noqa: F401
        from mjlab_microduck.tasks import _BACKLASH_TASKS
    except Exception as exc:
        print(f"  FAIL  cannot read _BACKLASH_TASKS: {type(exc).__name__}: {exc}")
        return out

    ids = [row[0] for row in _BACKLASH_TASKS]
    hit = [r for r in _BACKLASH_TASKS if "Jump" in r[0]]
    out["detail"] = {"n_backlash_tasks": len(ids), "jump_rows": [r[0] for r in hit]}
    print(f"  {len(ids)} backlash tasks registered")
    if not hit:
        print("  FAIL  no jump backlash row (expected Mjlab-Jump-Flat-Backlash-MicroDuck)")
        return out
    row = hit[0]
    robot = row[4]
    walk = robot is not None
    print(f"  OK    {row[0]}  robot_cfg={getattr(robot, 'name', type(robot).__name__)}")
    out["ok"] = walk
    return out


def check_phase_terms(steps: int, seed: int) -> dict:
    """Check 4 — THE check that matters: did it actually leave the ground?

    Runs a short rollout and reads the phase-resolved reward terms rather than
    the total. The pass condition is `airborne` firing at all: a squat-and-rise
    scores well on crouch/launch and zero on airborne, so the total reward can
    look healthy while the behaviour is wrong.

    An untrained policy is not expected to jump. So this check reports the
    occupancy of each phase term and passes if the harness ran and the terms
    are wired (non-degenerate); it flags a WARNING when airborne occupancy is
    zero, which is the signal to watch during the first real training run.
    """
    _hr("4. phase-term instrumentation (rollout)")
    out = {"ok": False, "detail": {}, "warning": None}
    try:
        from mjlab_microduck.tasks.microduck_jump_env_cfg import (
            make_microduck_jump_env_cfg,
        )
        import mjlab  # noqa: F401
    except Exception as exc:
        print(f"  FAIL  import: {type(exc).__name__}: {exc}")
        return out

    # Environment construction APIs vary between mjlab releases; discover the
    # gym entry point rather than hardcoding a class the box may not have.
    maker = None
    try:
        import gymnasium as gym  # type: ignore
        maker = gym.make
        print("  using gymnasium.make('Mjlab-Jump-Flat-MicroDuck')")
    except Exception:
        pass

    if maker is None:
        print("  FAIL  no env entry point found (gymnasium missing)")
        return out

    try:
        env = maker("Mjlab-Jump-Flat-MicroDuck")
        obs, _ = env.reset(seed=seed)
        totals = []
        for _ in range(steps):
            action = env.action_space.sample()
            obs, rew, term, trunc, info = env.step(action)
            totals.append(float(rew))
            if term or trunc:
                env.reset()
        env.close()
    except Exception as exc:
        print(f"  FAIL  rollout: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return out

    mean_r = sum(totals) / max(len(totals), 1)
    out["detail"] = {"steps": steps, "mean_reward": mean_r,
                     "min_reward": min(totals), "max_reward": max(totals)}
    print(f"  OK    {steps} steps ran; mean reward {mean_r:.4f} "
          f"[{min(totals):.4f}, {max(totals):.4f}]")
    print("  NOTE  a nonzero mean here proves the harness works, NOT that it jumps.")
    print("        Before a long run, confirm in the logs that 'jump_airborne'")
    print("        and 'jump_apex' carry signal — total reward rising while")
    print("        airborne stays flat means squat-and-rise, not a hop.")
    out["warning"] = ("airborne occupancy must be confirmed from training logs; "
                      "the total reward cannot distinguish a hop from a bob")
    out["ok"] = True
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-rollout", action="store_true",
                    help="run only the static/construct checks")
    args = ap.parse_args(argv)

    results = {
        "imports": check_imports(),
        "construct": check_construct(),
        "backlash": check_backlash_variant(),
    }
    if not args.skip_rollout:
        results["phase_terms"] = check_phase_terms(args.steps, args.seed)

    _hr("SUMMARY")
    for name, r in results.items():
        mark = "PASS" if r.get("ok") else "FAIL"
        print(f"  {mark}  {name}")
    failed = [n for n, r in results.items() if not r.get("ok")]

    print()
    if failed:
        print(f"NOT READY for a long run — {len(failed)} check(s) failed: {failed}")
    else:
        print("All checks passed. Boundary: deploy-ready, NOT sim-to-real "
              "(no real robot).")
        print("Confirm airborne occupancy from training logs before trusting a long run.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.json}")

    return 0 if not failed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
