from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image


def _safe_name(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in ("-", "_", ".") else "_"
        for character in value.strip()
    )
    return safe or "pledge"


def _session_dirs_for_pledge(runtime_dir: Path, pledge_id: str) -> list[Path]:
    session_dirs: list[Path] = []
    for candidate in runtime_dir.iterdir():
        state_path = candidate / "state.json"
        if not candidate.is_dir() or candidate.name == "_pledges" or not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(state.get("pledge_id") or "") == pledge_id:
            session_dirs.append(candidate)
    return session_dirs


def _optimize_png(path: Path) -> tuple[int, int]:
    original_stat = path.stat()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with Image.open(path) as source:
            source.load()
            original_mode = source.mode
            original_size = source.size
            original_pixels = source.tobytes()
            save_options = {
                "format": "PNG",
                "optimize": True,
                "compress_level": 9,
            }
            if source.info.get("icc_profile"):
                save_options["icc_profile"] = source.info["icc_profile"]
            if source.info.get("dpi"):
                save_options["dpi"] = source.info["dpi"]
            source.save(temporary, **save_options)

        with Image.open(temporary) as optimized:
            optimized.load()
            pixels_equal = (
                optimized.mode == original_mode
                and optimized.size == original_size
                and optimized.tobytes() == original_pixels
            )
        optimized_size = temporary.stat().st_size
        if pixels_equal and optimized_size < original_stat.st_size:
            os.chmod(temporary, original_stat.st_mode)
            os.utime(
                temporary,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            os.replace(temporary, path)
            return original_stat.st_size, optimized_size
        return original_stat.st_size, original_stat.st_size
    finally:
        temporary.unlink(missing_ok=True)


def compress_pledge_artifacts(runtime_dir: Path, pledge_id: str) -> dict[str, int]:
    roots = _session_dirs_for_pledge(runtime_dir, pledge_id)
    media_dir = runtime_dir / "_pledges" / "media" / _safe_name(pledge_id)
    if media_dir.is_dir():
        roots.append(media_dir)

    files = sorted(
        {
            path
            for root in roots
            for path in root.rglob("*.png")
            if path.is_file()
        }
    )
    before_bytes = 0
    after_bytes = 0
    optimized_files = 0
    failed_files = 0
    for path in files:
        try:
            before, after = _optimize_png(path)
        except Exception as exc:  # noqa: BLE001
            failed_files += 1
            print(f"[ArtifactCompression] Could not optimize {path}: {exc}")
            continue
        before_bytes += before
        after_bytes += after
        if after < before:
            optimized_files += 1

    return {
        "scanned_files": len(files),
        "optimized_files": optimized_files,
        "failed_files": failed_files,
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "saved_bytes": before_bytes - after_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--pledge-id", required=True)
    args = parser.parse_args()
    try:
        os.nice(15)
    except OSError:
        pass
    result = compress_pledge_artifacts(args.runtime_dir.resolve(), args.pledge_id)
    print(f"[ArtifactCompression] {args.pledge_id}: {json.dumps(result, sort_keys=True)}")


if __name__ == "__main__":
    main()
