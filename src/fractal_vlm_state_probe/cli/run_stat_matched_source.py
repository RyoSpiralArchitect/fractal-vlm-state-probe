"""Preparation, separately approved capture, and zero-inference saved-data checks."""

import argparse
import subprocess
from pathlib import Path

from fractal_vlm_state_probe.stat_matched_source_capture import acquire
from fractal_vlm_state_probe.stat_matched_source_native import (
    NativeBackend,
    qualification,
)
from fractal_vlm_state_probe.stat_matched_source_storage import (
    read,
    sha,
    write,
)
from fractal_vlm_state_probe.stat_matched_source_verify import (
    analyze_verified,
    verify_capture,
)

REPO = Path(__file__).resolve().parents[3]
PROFILE = "configs/stat_matched_source_capture_preparation_v1.json"


def require_release(repo, path, publication=None):
    if path is None:
        raise ValueError(
            "separate explicit acquisition release required before runtime preparation"
        )
    release = read(path)
    profile = read(repo / PROFILE)
    if (
        release.get("study_id") != profile["required_separate_release_id"]
        or release.get("approved_for_execution") is not True
        or release.get("preparation_sha256") != sha(repo / PROFILE)
    ):
        raise ValueError("separate approved release binding differs")
    canonical = repo / "configs" / (profile["required_separate_release_id"] + ".json")
    if Path(path).resolve() != canonical.resolve():
        raise ValueError("canonical separately committed acquisition release required")
    relative = str(canonical.relative_to(repo))
    release_commit = (
        subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", relative], cwd=repo
        )
        .decode()
        .strip()
    )
    if (
        not release_commit
        or subprocess.check_output(
            ["git", "show", f"{release_commit}:{relative}"], cwd=repo
        )
        != canonical.read_bytes()
    ):
        raise ValueError("commit the separate acquisition release before execution")
    if publication is None:
        raise ValueError("public merged release receipt required")
    published_release = read(publication)
    if (
        published_release["state"] != "MERGED"
        or not published_release["url"].startswith(
            "https://github.com/RyoSpiralArchitect/fractal-vlm-state-probe/pull/"
        )
        or not published_release["mergedAt"]
    ):
        raise ValueError("public acquisition-release receipt differs")
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            release_commit,
            published_release["mergeCommit"]["oid"],
        ],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            published_release["mergeCommit"]["oid"],
            "HEAD",
        ],
        cwd=repo,
        check=True,
    )
    spec = read(repo / "configs/stat_matched_source_panel_v1.json")
    previous = read(repo / spec["preserve_inherited_frozen_hashes_from"])
    core_profile = read(repo / "configs/stat_matched_source_core_v1.json")
    names = (
        set(previous["code_sha256"])
        | set(core_profile["registration_sha256"])
        | set(profile["preserved_core_sha256"])
        | set(profile["implementation_files"])
    )
    if set(release["code_sha256"]) != names:
        raise ValueError("release does not freeze the complete implementation")
    commit = release["implementation_commit"]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, release_commit],
        cwd=repo,
        check=True,
    )
    for name in names:
        p = repo / name
        frozen = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo)
        import hashlib

        if sha(p) != release["code_sha256"][name] or hashlib.sha256(
            frozen
        ).hexdigest() != sha(p):
            raise ValueError(
                "commit the complete reviewed implementation before acquisition"
            )
    receipt = Path(release["public_receipt"])
    if not receipt.is_absolute():
        receipt = repo / receipt
    public = read(receipt)
    if (
        sha(receipt) != release["public_receipt_sha256"]
        or public["state"] != "MERGED"
        or not public["url"].startswith(
            "https://github.com/RyoSpiralArchitect/fractal-vlm-state-probe/pull/"
        )
    ):
        raise ValueError("public merged-PR receipt differs")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, public["mergeCommit"]["oid"]],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", public["mergeCommit"]["oid"], "HEAD"],
        cwd=repo,
        check=True,
    )
    return release


def run(
    mode,
    output,
    *,
    release=None,
    publication=None,
    preflight=None,
    capture=None,
    first=None,
    second=None,
):
    output = Path(output).resolve()
    profile = read(REPO / PROFILE)
    if mode == "analyze":
        if capture is None or first is None or second is None:
            raise ValueError("capture and two verified roots required")
        return analyze_verified(Path(capture), output, Path(first), Path(second))
    if mode == "verify":
        if capture is None:
            raise ValueError("capture root required")
        return verify_capture(Path(capture), output)
    approved = (
        require_release(REPO, release, publication)
        if release is not None or mode == "capture"
        else None
    )
    key = (
        ("preparation_root" if approved is None else "preflight_root")
        if mode == "preflight"
        else "capture_root"
    )
    if (
        mode not in ("preflight", "capture")
        or output != (REPO / profile[key]).resolve()
        or output.exists()
    ):
        raise ValueError("canonical fresh root required; no retry/resume")
    import shutil

    spec = read(REPO / "configs/stat_matched_source_panel_v1.json")
    if mode == "preflight":
        output.mkdir(parents=True)
    try:
        free = shutil.disk_usage(output.parent).free
        if free < spec["budget"]["required_launch_free_bytes"]:
            raise ValueError("registered 28 GiB launch gate not met")
        frozen = qualification(REPO)
    except BaseException as error:
        if mode == "preflight":
            write(
                output / "preflight_failure.json",
                {
                    "status": "FAILED_PREFLIGHT_RETAINED",
                    "type": type(error).__name__,
                    "message": str(error),
                    "new_processor_calls": 0,
                    "new_model_contexts": 0,
                    "model_weights_loaded": False,
                },
            )
        raise
    frozen["execution_release_sha256"] = None if approved is None else sha(release)
    frozen["public_release_receipt_sha256"] = (
        None if approved is None else sha(publication)
    )
    frozen["preparation_profile_sha256"] = sha(REPO / PROFILE)
    if approved is not None:
        code_receipt_path = Path(approved["public_receipt"])
        if not code_receipt_path.is_absolute():
            code_receipt_path = REPO / code_receipt_path
        frozen["execution_release_data"] = approved
        frozen["execution_release_commit"] = (
            subprocess.check_output(
                [
                    "git",
                    "log",
                    "-1",
                    "--format=%H",
                    "--",
                    str(Path(release).resolve().relative_to(REPO)),
                ],
                cwd=REPO,
            )
            .decode()
            .strip()
        )
        frozen["public_release_receipt_text"] = (
            Path(publication).read_bytes().decode("utf-8")
        )
        frozen["public_code_receipt_text"] = code_receipt_path.read_bytes().decode(
            "utf-8"
        )
    frozen["implementation_sha256"] = (
        {} if approved is None else approved["code_sha256"]
    )
    if mode == "preflight":
        write(output / "frozen.json", frozen)
        result = {
            "status": "PREPARED_NOT_RELEASED"
            if approved is None
            else "PASS_FOR_APPROVED_RELEASE",
            "frozen_sha256": sha(output / "frozen.json"),
            "new_processor_calls": 0,
            "model_weights_loaded": False,
            "new_model_contexts": 0,
            "available_bytes": free,
        }
        write(output / "preflight.json", result)
        return result
    if preflight is None:
        raise ValueError("approved preflight required")
    preflight = Path(preflight).resolve()
    pre = read(preflight / "preflight.json")
    if (
        preflight != (REPO / profile["preflight_root"]).resolve()
        or pre["status"] != "PASS_FOR_APPROVED_RELEASE"
        or pre["model_weights_loaded"]
        or pre["new_processor_calls"]
        or pre["new_model_contexts"]
        or pre["available_bytes"] < spec["budget"]["required_launch_free_bytes"]
        or (preflight / "preflight_failure.json").exists()
        or pre["frozen_sha256"] != sha(preflight / "frozen.json")
        or frozen != read(preflight / "frozen.json")
    ):
        raise ValueError("approved preflight changed or incomplete")
    frozen["outer_preflight"] = {
        "receipt_text": (preflight / "preflight.json").read_bytes().decode("utf-8"),
        "receipt_sha256": sha(preflight / "preflight.json"),
        "frozen_text": (preflight / "frozen.json").read_bytes().decode("utf-8"),
    }
    return acquire(output, frozen, NativeBackend(frozen))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "capture", "verify", "analyze"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--public-release-receipt", type=Path)
    parser.add_argument("--preflight-root", type=Path)
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--first-verified-root", type=Path)
    parser.add_argument("--second-verified-root", type=Path)
    args = parser.parse_args()
    result = run(
        args.mode,
        args.output_root,
        release=args.release,
        publication=args.public_release_receipt,
        preflight=args.preflight_root,
        capture=args.capture_root,
        first=args.first_verified_root,
        second=args.second_verified_root,
    )
    print(result["status"])


if __name__ == "__main__":
    main()
