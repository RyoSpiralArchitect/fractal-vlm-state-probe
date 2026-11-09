"""Separate preparation, native acquisition and ordered saved-data verification."""

import argparse
from pathlib import Path

from fractal_vlm_state_probe.qwen_calibration_admission import admit, preflight
from fractal_vlm_state_probe.stat_matched_source_storage import inside, inventory, read, write


def run(repo, action):
    repo = Path(repo).resolve()
    if action == "preflight":
        return preflight(repo)
    plan, jobs, provenance = admit(repo, for_capture=action == "capture")
    capture_root = inside(repo, plan["future_native_root"])
    roots = [inside(repo, p) for p in plan["future_verification_roots"]]
    source = inside(repo, plan["capture_root"])
    if action == "capture":
        if capture_root.exists() or any(p.exists() for p in roots):
            raise FileExistsError("consumed acquisition or verification path; no retry")
        from fractal_vlm_state_probe.qwen_calibration_capture import capture
        from fractal_vlm_state_probe.qwen_calibration_native import NativeBackend

        # Freeze acquisition failure even if loading fails before the first context.
        try:
            return capture(capture_root, plan, jobs, source, lambda: NativeBackend(
                provenance["environment"]["snapshot"],
                read(source / "frozen.json")["source_prompt"], plan), provenance)
        except Exception as error:
            capture_root.mkdir(parents=True, exist_ok=True)
            if not (capture_root / "failure.json").exists():
                write(capture_root / "failure.json", {"error": str(error), "retry_authorized": False})
            raise
    from fractal_vlm_state_probe.qwen_calibration_verify import verify_to

    if action == "verify":
        if roots[1].exists():
            raise FileExistsError("verification replay already consumed")
        return verify_to(capture_root, roots[0], plan, jobs, source, provenance=provenance)
    if action != "verify-replay":
        raise ValueError("unknown action")
    if set(p.name for p in roots[0].iterdir()) != {"summary.json"}:
        raise ValueError("first verification must succeed before replay")
    result = verify_to(capture_root, roots[1], plan, jobs, source, provenance=provenance)
    try:
        if inventory(roots[0]) != inventory(roots[1]):
            raise ValueError("ordered verifier trees differ")
    except Exception as error:
        write(roots[1] / "failure.json", {"error": str(error), "retry_authorized": False})
        raise
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "capture", "verify", "verify-replay"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(run(args.repo, args.action))
