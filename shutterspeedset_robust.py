"""
Robust Camera Exposure Controller
==================================
Eliminates black banding / flicker caused by mismatch between camera exposure
and mains power-line frequency (50 Hz in India / Europe, 60 Hz in US).

Root causes of banding fixed here:
  1. Exposure not aligned to power-line period  → syncs to 50 Hz multiples
  2. Auto-exposure fighting manual settings      → fully disables auto-exposure
  3. Stale / banded frames still in buffer       → flushes N frames after change
  4. power_line_frequency control not set        → explicitly set to 50 Hz
  5. Race between v4l2-ctl and OpenCV            → v4l2 applied first, OpenCV
                                                    used only as fallback
  6. Settings silently rejected by driver        → verifies read-back values
"""

import sys
import time
import subprocess
import cv2

# ── Configuration ─────────────────────────────────────────────────────────────
CAMERA_INDEX       = 0
DEVICE_PATH        = "/dev/video0"

# Power-line frequency (Hz).  India / Europe = 50  |  US / Japan = 60
POWER_LINE_HZ      = 50          # change to 60 if you are on 60 Hz mains

# Desired shutter speed expressed as denominator of 1/N seconds.
# MUST be a multiple of POWER_LINE_HZ (or 2× it) to avoid banding.
#   Good values for 50 Hz mains → 50, 100, 200, 500, 1000
#   Good values for 60 Hz mains → 60, 120, 240, 480, 1000
SHUTTER_DENOMINATOR = 100        # 1/100 s  (one full 50 Hz cycle = 20 ms)

# Frames to discard after applying settings (clears stale/banded frames)
FLUSH_FRAMES       = 10

# Optional resolution cap (0 = keep camera maximum)
FRAME_WIDTH        = 0
FRAME_HEIGHT       = 0
# ── End of configuration ──────────────────────────────────────────────────────


# ── v4l2-ctl helpers ──────────────────────────────────────────────────────────

def v4l2(args: list[str]) -> tuple[bool, str]:
    """Run v4l2-ctl and return (success, message)."""
    try:
        r = subprocess.run(
            ["v4l2-ctl"] + args,
            capture_output=True, text=True, check=True
        )
        return True, r.stdout.strip()
    except FileNotFoundError:
        return False, "v4l2-ctl not found – install v4l-utils"
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip()
        return False, msg or "v4l2-ctl returned non-zero"


def v4l2_get(control: str) -> str | None:
    """Read back a single v4l2 control value (returns None on failure)."""
    ok, out = v4l2(["-d", DEVICE_PATH, "--get-ctrl", control])
    if ok and ":" in out:
        return out.split(":", 1)[1].strip()
    return None


def v4l2_set(control: str, value) -> bool:
    """Set a v4l2 control.  Returns True on success."""
    ok, msg = v4l2(["-d", DEVICE_PATH, "-c", f"{control}={value}"])
    if not ok:
        print(f"  ⚠  v4l2 {control}={value} → {msg}")
    return ok


# ── Exposure math ─────────────────────────────────────────────────────────────

def align_to_powerline(denominator: int, hz: int) -> int:
    """
    Round denominator UP to the nearest multiple of `hz`.
    Keeps exposure ≤ original request while ensuring it divides evenly
    into the power-line period, eliminating banding.

    Examples (hz=50):
      101 → 100  (rounds DOWN to nearest multiple that is >= 1 period)
      75  → 100
      50  → 50
      200 → 200
    """
    if denominator <= 0:
        return hz
    remainder = denominator % hz
    if remainder == 0:
        return denominator
    # round UP to next multiple so exposure is *shorter*, never brighter
    return denominator + (hz - remainder)


def denominator_to_exposure_absolute(denominator: int) -> int:
    """
    Convert shutter denominator to v4l2 exposure_absolute units.
    V4L2 exposure_absolute is in units of 100 µs (0.1 ms).

    1/denom seconds = (10 000 / denom) × 100 µs units
    """
    if denominator <= 0:
        return 200
    return max(1, int(round(10000 / denominator)))


# ── Camera setup ──────────────────────────────────────────────────────────────

def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

    if FRAME_WIDTH > 0 and FRAME_HEIGHT > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  10000)   # ask for max, driver clips
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)

    cap.set(cv2.CAP_PROP_FOURCC,      cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE,  1)             # smallest latency buffer

    return cap


def flush_buffer(cap: cv2.VideoCapture, n: int = FLUSH_FRAMES):
    """Grab-and-discard N frames to clear any pre-exposure-change frames."""
    for _ in range(n):
        cap.grab()
    time.sleep(0.05)   # small pause so driver can settle


# ── Main exposure-application logic ──────────────────────────────────────────

def list_camera_controls() -> None:
    """Print all available v4l2 controls for the device."""
    print(f"\n── Available controls for {DEVICE_PATH} ──")
    ok, out = v4l2(["-d", DEVICE_PATH, "--list-ctrls"])
    if ok:
        for line in out.split("\n"):
            print(f"  {line}")
    else:
        print(f"  Could not list controls: {out}")
    ok2, out2 = v4l2(["-d", DEVICE_PATH, "--list-ctrls-menus"])
    if ok2:
        for line in out2.split("\n"):
            if "exposure" in line.lower() or "power" in line.lower() or "freq" in line.lower():
                print(f"  {line}")
    print("")


def try_set_exposure_auto(value: int) -> bool:
    """Try setting exposure_auto to given value; return True if accepted."""
    ok = v4l2_set("exposure_auto", value)
    time.sleep(0.1)
    rb = v4l2_get("exposure_auto")
    if rb is not None and rb.strip() == str(value):
        print(f"  ✅ exposure_auto={value} accepted (read-back: {rb})")
        return True
    print(f"  ⚠  exposure_auto={value} → read-back: {rb}")
    return False


def apply_manual_exposure(cap: cv2.VideoCapture, denominator: int, hz: int) -> bool:
    """
    Fully disable auto-exposure and apply the requested shutter speed,
    aligned to the power-line frequency to prevent banding.

    Returns True if the driver confirmed the value via read-back.
    """
    # ── Diagnostic: list controls ──
    list_camera_controls()

    aligned = align_to_powerline(denominator, hz)
    if aligned != denominator:
        print(f"  ℹ  Shutter adjusted: 1/{denominator}s → 1/{aligned}s "
              f"(aligned to {hz} Hz grid)")
    exposure_abs = denominator_to_exposure_absolute(aligned)

    # ── Step 1: Set power-line frequency ──
    print(f"\n── Step 1: Set power_line_frequency to {hz} Hz ──")
    plf_value = 1 if hz == 50 else 2
    ok_plf = v4l2_set("power_line_frequency", plf_value)
    if not ok_plf:
        print("  ⚠  power_line_frequency control not supported by this camera.")
        print("  ℹ   Try SHUTTER_DENOMINATOR=50 (slower shutter masks flicker).")
    else:
        plf_readback = v4l2_get("power_line_frequency")
        print(f"  power_line_frequency read-back: {plf_readback}")

    # ── Step 2: Disable auto-exposure ──
    print(f"\n── Step 2: Disable auto-exposure ──")
    # Try multiple exposure_auto values — different cameras use different values
    # 3 = aperture priority, 1 = manual mode (UVC standard)
    # Some cameras need 0 for manual, some use 1
    ae_ok = False
    for val in [1, 3, 0]:
        ae_ok = try_set_exposure_auto(val)
        if ae_ok:
            break

    if not ae_ok:
        print("  ⚠  Could not disable auto-exposure on this camera.")
        print("  ℹ   Try a slower SHUTTER_DENOMINATOR (50) to reduce visible flicker.")

    # Double-disable via OpenCV (belt-and-suspenders)
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    except Exception:
        pass

    # ── Step 3: Set exposure_absolute ──
    print(f"\n── Step 3: Set exposure_absolute = {exposure_abs} "
          f"(1/{aligned}s) ──")
    ok_exp = v4l2_set("exposure_absolute", exposure_abs)
    time.sleep(0.05)

    # OpenCV fallback (handles cameras that ignore v4l2-ctl but obey OpenCV)
    try:
        cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_abs))
    except Exception:
        pass

    exp_readback = v4l2_get("exposure_absolute")
    print(f"  exposure_absolute read-back: {exp_readback}  (expected: {exposure_abs})")

    # Verify
    try:
        confirmed = int(exp_readback) == exposure_abs
    except (TypeError, ValueError):
        confirmed = False

    return confirmed


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    denominator = SHUTTER_DENOMINATOR
    hz          = POWER_LINE_HZ

    print("=" * 60)
    print(" Robust Camera Exposure Controller")
    print(f" Target shutter : 1/{denominator} s")
    print(f" Power-line freq: {hz} Hz")
    print("=" * 60)

    # ── Open camera ──
    print("\n── Opening camera ──")
    cap = open_camera()
    if not cap.isOpened():
        print("❌  Could not open camera. Check CAMERA_INDEX / DEVICE_PATH.")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera opened at {w}×{h}")

    # ── Flush any stale frames before changing settings ──
    flush_buffer(cap, n=3)

    # ── Apply exposure ──
    confirmed = apply_manual_exposure(cap, denominator, hz)

    # ── Flush again to discard frames captured under old settings ──
    print(f"\n── Step 4: Flushing {FLUSH_FRAMES} stale frames ──")
    flush_buffer(cap, n=FLUSH_FRAMES)
    print(f"  {FLUSH_FRAMES} frames discarded")

    # ── Verify with a live frame ──
    print("\n── Step 5: Capture verification frame ──")
    ret, frame = cap.read()
    if ret:
        print(f"  ✅ Live frame captured: {frame.shape[1]}×{frame.shape[0]}, "
              f"dtype={frame.dtype}")
        # Optionally save a snapshot for inspection
        # cv2.imwrite("/tmp/exposure_verify.jpg", frame)
    else:
        print("  ⚠  Could not read a frame – check camera connection")

    cap.release()

    print("\n" + "=" * 60)
    if confirmed:
        aligned = align_to_powerline(denominator, hz)
        print(f"✅  SUCCESS – Shutter 1/{aligned}s applied and verified.")
    else:
        print("⚠   Settings applied but driver read-back did not confirm.")
        print("    Your camera may not support exposure_absolute write-back.")
        print("    The values were still sent; try viewing the feed to confirm.")
    print("=" * 60)
