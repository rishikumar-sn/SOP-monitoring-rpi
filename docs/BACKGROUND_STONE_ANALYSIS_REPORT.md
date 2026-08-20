# Background Stone Analysis, Results Viewer & Honest Timers — Implementation Report

Date: 2026-08-19 · Target: SOP-monitoring-rpi (Raspberry Pi, Hailo-8, localhost)

## Files modified

| File | Change |
|---|---|
| `integrated_ui_app.py` | +~455 lines: background stone job infrastructure, honest timing history (median of last 5 runs), forced Half Cut mode, result-preserving optimizations |
| `webui/main.html` | +~386 lines: Results viewer overlay, honest jewellery-analysis timer, background stone-job banner/states, Half Cut-only UI |

## Architecture

### Background Stone Analysis (eligible branches)
- `ThreadPoolExecutor(max_workers=1, thread_name_prefix="stone-analysis")` + in-memory job registry holding the decoded working image, stone candidate mask, ignore mask, calibration and weight inputs.
- `/api/stone-detection/main` (integrated_ui_app.py:10764) queues the job instead of blocking when the branch is `segmentation` or `direct_stone` AND the jewel type is not stone-weight-exempt. Response returns immediately with status "Stone analysis started in the background...".
- Worker `_run_stone_job_worker` (line 6088): runs `run_stone_pipeline` WITHOUT `STATE_LOCK` (compute is off the UI thread), then merges the result into `state["stone_detection"]["main"]` under `STATE_LOCK` — but only if `_stone_job_context_matches` (session_id, pledge_id, jewel_index, working image path all unchanged, job not cancelled).
- Result lands in the same `stone_detection.main` object the PDF/report generator already reads — the report is byte-identical to the foreground path.
- **Eligible** (background): Haram, Necklace, Dollar chain, Kasu Mala — in segmentation and direct-stone branches.
- **Excluded** (stay foreground): Bangle, Finger ring/Ring, Earing/Jumkha, Earrings/Jhumki (stone-weight exempt) and the dimension branch (side-stone only, no main-image analysis).
- Acid Test starts as soon as the job is queued: `purity_upstream_ready` returns true for queued/running jobs; the frontend auto-advances to the purity stage after submitting the job.
- **Gating while a job runs**: `build_final_summary` still requires `stones.main` or a skip, so Final/Next Jewel/pledge completion and PDF generation wait for the job; `/api/generate-pdf` returns 409 "Waiting for Stone Analysis to finish..." while a job is active.
- **Invalidation** (job cancelled + result discarded even if already computed): re-running jewellery analysis, stage skips, stone settings, background calibration, learned profile changes, classify confirm, reset, next jewel, new source capture.
- **Restart**: `_reconcile_stone_job_after_restart` marks an in-memory job as failed ("Interrupted by an application restart. Re-run the stone analysis.") so the UI never waits forever on a ghost job.
- Worker merge deliberately does NOT call `reset_purity_state` — an in-progress or completed acid test is preserved. The synchronous fallback path keeps the original `reset_purity_state` behaviour unchanged.

### Honest timers
- Backend records real durations in `runtime_sessions/analysis_timing.json` (`TIMING_HISTORY_MAX_RUNS=12`, median of the last 5 runs used as the estimate) for `jewellery_analysis` and `stone_analysis`; estimates are exposed in every state snapshot via `state["timing_estimates"]`.
- Jewellery Analysis (foreground): loading overlay shows "Elapsed MM:SS · ~MM:SS remaining", or "Estimating completion time..." on the first run, "Finishing analysis..." when the estimate is exceeded.
- Stone Analysis (background): the purity-panel banner and the Stone Analysis panel show elapsed time computed from the job's server `started_at`, plus the same ETA logic; both refresh on the 900 ms poll.

### Half Cut (front-only) is the ONLY stone-weight mode
- Backend: both `/api/stone-detection/main` and `/api/side-stones/run` force `STONE_SETTING_PROFILE_FRONT_ONLY`; the client payload is ignored entirely, so Full Cut / Unknown cannot be selected even by an older frontend.
- Frontend: the Full Cut/Unsure chip row was removed; the hidden selects expose only the `front_only_shallow` option. Inert CSS/JS for chips remains but matches zero DOM elements.
- Unchanged: `FRONT_ONLY_AREAL_MASS_G_PER_MM2 = 0.001695`, uncertainty 0.15, `calibrate_weight_estimate_to_jewel_weight`, HSV/SAHI/FastSAM/thresholds, super-resolution — no accuracy-affecting constants touched.

### Result-preserving optimizations only
1. `run_stone_pipeline(..., image_bgr=...)` — the request thread already decoded the working image; the pipeline now accepts it instead of re-reading from disk (`cv2.imread` only when `None`). Removes one full-image decode per run.
2. `_normalized_binary_mask` — identity shortcut: when the mask is already on the target grid, binary (max ≤ 1) and returned as-is without a full-array copy + re-threshold. Verified all downstream consumers only read the mask (threshold with `> 0`, no in-place mutation): `extract_jewel_candidates`, `build_candidates_from_component_mask`, `append_candidate_from_mask`.
3. Background execution itself: the UI no longer blocks the RPi during stone analysis (the longest stage), while the computation is unchanged.

### Results viewer (frontend only)
- "Results" header button opens a full overlay with tabs (Weight / Jewellery Analysis / Stone Analysis) reusing the existing artifact-strip carousel. It shows only result images that exist in the current session state, auto-refreshes while open, closes via Escape, backdrop click or "Back to Workflow".

## Regression comparison (behavioural)

| Scenario | Before | After |
|---|---|---|
| Haram/Necklace/Dollar chain/Kasu Mala stone analysis | UI blocked ~20-120 s, no ETA, then auto-advance | Job starts instantly, honest elapsed+ETA banner, auto-advance to purity immediately |
| Bangle/Finger ring/Earrings stone analysis | foreground synchronous | unchanged (synchronous foreground) |
| Acid Test start | after stone analysis finished | immediately after job submission |
| Final/Next Jewel/PDF | after stone analysis | still after stone analysis (job must complete) |
| PDF while job running | would include missing stone data | 409 guard with clear message |
| App restart mid-analysis | (previously no background job existed) | job marked failed, re-run from Stone Analysis |
| Full Cut / Unsure selection | available | impossible (backend enforces Half Cut) |
| Jewellery analysis progress | spinner only | honest elapsed + median-of-5 ETA |
| Results review | scattered panels | Results button overlay with all current-session result images |

## Before/after timings (to be measured on device)
| Stage | Before | After |
|---|---|---|
| Jewellery analysis (segmentation pipeline) | unchanged | unchanged (compute identical; elapsed/ETA now shown) |
| Stone analysis — UI request round-trip | equals pipeline duration (~20-120 s) | < 1 s (queued immediately) |
| Stone analysis — total wall time | identical | identical (same pipeline, now off the UI thread) |

## Verification performed (static)
- `python3 -m py_compile integrated_ui_app.py` — OK
- Full JS syntax parse of the inline script (esprima; only pre-existing `??` operators needed normalization) — OK
- HTML div balance 231/231 — OK
- 45-point static checklist (endpoint branches, worker merge/validation/cancel, gating, invalidation sites, PDF guard, restart reconcile, timer wiring, panel states, Half Cut enforcement) — all pass
- No leftover `open_back_faceted` / `unknown` options in frontend; no stray `payload` reads in rewritten endpoints

## Real-device test checklist (16)

1. Haram → capture → classify → weight → jewellery analysis (note elapsed/ETA; second run should show ETA from history).
2. Stone Analysis on Haram: confirm background banner appears, elapsed ticks, acid test reachable immediately.
3. Start acid test while stone job runs; job completes mid-test — purity state must be preserved, main stone result appears, banner switches to success.
4. After job completes: Final panel enables, report PDF contains stone weight, stones count, stone percentage, gallery image.
5. Click Results button — tabs show Weight, Jewellery Analysis, Stone Analysis images; carousel navigates; closes via Escape/backdrop/Back to Workflow.
6. Open Results overlay, re-run weight capture — overlay auto-refreshes.
7. First-ever run (no timing history): both timers show "Estimating completion time...".
8. Run jewellery analysis 2+ times; second run's ETA should approximate the first run's actual duration (median of last 5).
9. Necklace + Kasu Mala + Dollar chain: same background behaviour as Haram (branch segmentation).
10. Bangle: stone analysis stays foreground (blocking overlay), side pipeline unaffected.
11. Finger ring / Earrings: stone analysis stays foreground (weight-exempt).
12. Click Next Jewel while a stone job runs: must be blocked (final not ready); complete the job, then Next Jewel works.
13. Re-capture source image / click Reset while a job runs: job cancelled, no stale result merges, no crash.
14. Restart the app (kill + relaunch) with a job running: banner shows "Interrupted by an application restart. Re-run the stone analysis."
15. Try to force Full Cut via modified frontend payload: backend still reports Half Cut (front-only) weight.
16. Generate PDF while a job is queued/running: 409 message shown; after completion, PDF includes the stone section.

## Notes
- Timing history lives in `runtime_sessions/analysis_timing.json`; delete it to reset estimates.
- `speak()` is thread-safe (TTS queue); the worker speaks "Stone analysis completed." after a successful merge.
- No pytest suite exists in this repo; verification is static + the on-device checklist above.