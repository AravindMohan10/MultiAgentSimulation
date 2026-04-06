"""
Synthetic data tests for Phase 1 metrics.

Run from repo root:
  PYTHONPATH=. python3 tests/test_phase1_metrics.py

Uses repo root resolved from this file (no hard-coded absolute path).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.metrics.beacon_behavior import detect_beacon_behavior
from src.metrics.information_threshold import analyze_information_threshold
from src.metrics.phase_transition import detect_phase_transition
from src.metrics.superadditivity import compute_superadditivity


def _synth_step(
    i: int,
    *,
    hero_xy: tuple[float, float] = (150.0, 150.0),
    v1_xy: tuple[float, float] = (100.0, 100.0),
    v2_xy: tuple[float, float] = (50.0, 50.0),
    v2_target: tuple[float, float],
    v1_sees_hero: bool,
    v2_blind: bool,
    v2_steps_since: int,
) -> dict:
    """Minimal step record matching engine JSONL (beacon + phase metrics)."""
    return {
        "step_index": i,
        "hero_position": [hero_xy[0], hero_xy[1], 0.0],
        "villain_positions": {
            "villain_1": [v1_xy[0], v1_xy[1], 0.0],
            "villain_2": [v2_xy[0], v2_xy[1], 0.0],
        },
        "per_agent": [
            {
                "agent_id": "hero_1",
                "role": "hero",
                "actual_position": [hero_xy[0], hero_xy[1]],
                "actual_movement": [0.5, 0.5],
                "hero_truly_visible": None,
                "steps_since_hero_seen": None,
                "llm_target_position": [155.0, 155.0],
            },
            {
                "agent_id": "villain_1",
                "role": "villain",
                "actual_position": [v1_xy[0], v1_xy[1]],
                "actual_movement": [0.8, 0.6],
                "hero_truly_visible": True if v1_sees_hero else False,
                "steps_since_hero_seen": 0 if v1_sees_hero else i + 10,
                "llm_target_position": [155.0, 155.0],
            },
            {
                "agent_id": "villain_2",
                "role": "villain",
                "actual_position": [v2_xy[0], v2_xy[1]],
                "actual_movement": [0.7, 0.7],
                "hero_truly_visible": False if v2_blind else True,
                "steps_since_hero_seen": v2_steps_since,
                "llm_target_position": [float(v2_target[0]), float(v2_target[1])],
            },
        ],
    }


# ─────────────────────────────────────────────
# TEST 1: Beacon behavior — SHOULD DETECT
# ─────────────────────────────────────────────
def test_beacon_detected() -> dict:
    steps = []
    for i in range(30):
        steps.append(
            _synth_step(
                i,
                v2_target=(98.0, 98.0),
                v1_sees_hero=True,
                v2_blind=True,
                v2_steps_since=i + 5,
            )
        )
    result = detect_beacon_behavior(steps)
    assert result["beacon_detected"] is True, f"Expected beacon_detected=True, got {result}"
    assert result["beacon_duration"] >= 5, f"Expected duration>=5, got {result['beacon_duration']}"
    assert result["v1_visible_during_beacon"] > 0.8, (
        f"Expected v1_visible>0.8, got {result['v1_visible_during_beacon']}"
    )
    print("✓ TEST 1 PASSED: Beacon detected correctly")
    return result


# ─────────────────────────────────────────────
# TEST 2: Beacon behavior — SHOULD NOT DETECT
# ─────────────────────────────────────────────
def test_beacon_not_detected() -> dict:
    steps = []
    for i in range(30):
        steps.append(
            _synth_step(
                i,
                v2_target=(20.0, 20.0),
                v1_sees_hero=True,
                v2_blind=True,
                v2_steps_since=i + 5,
            )
        )
    result = detect_beacon_behavior(steps)
    assert result["beacon_detected"] is False, f"Expected beacon_detected=False, got {result}"
    print("✓ TEST 2 PASSED: No false beacon detection")
    return result


# ─────────────────────────────────────────────
# TEST 3: Beacon — v1 BLIND (theory of mind test)
# ─────────────────────────────────────────────
def test_beacon_v1_blind() -> dict:
    steps = []
    for i in range(30):
        steps.append(
            _synth_step(
                i,
                v2_target=(98.0, 98.0),
                v1_sees_hero=False,
                v2_blind=True,
                v2_steps_since=i + 5,
            )
        )
    result = detect_beacon_behavior(steps)
    if result["beacon_detected"]:
        assert result["v1_visible_during_beacon"] < 0.2, (
            f"Expected low v1_visible when v1 blind, got {result['v1_visible_during_beacon']}"
        )
        assert result["theory_of_mind_score"] < result["beacon_duration"] * 0.2, (
            "ToM score should be low when v1 is blind"
        )
    print("✓ TEST 3 PASSED: ToM score correctly low when v1 blind")
    return result


# ─────────────────────────────────────────────
# TEST 4: Superadditivity — SUPERADDITIVE case
# ─────────────────────────────────────────────
def test_superadditivity_positive() -> dict:
    episodes_2v = [
        {"first_contact_step_any": 34, "seed": 0, "num_villains": 2, "outcome": "hero_escaped"},
        {"first_contact_step_any": 38, "seed": 1, "num_villains": 2, "outcome": "hero_escaped"},
        {"first_contact_step_any": 31, "seed": 2, "num_villains": 2, "outcome": "hero_escaped"},
    ]
    episodes_1v = [
        {"first_contact_step_any": 87, "seed": 0, "num_villains": 1, "outcome": "hero_escaped"},
        {"first_contact_step_any": 92, "seed": 1, "num_villains": 1, "outcome": "hero_escaped"},
        {"first_contact_step_any": 85, "seed": 2, "num_villains": 1, "outcome": "hero_escaped"},
    ]
    result = compute_superadditivity(episodes_2v, episodes_1v)
    assert result["superadditivity_index"] is not None and result["superadditivity_index"] > 1.0, (
        f"Expected index > 1.0, got {result['superadditivity_index']}"
    )
    assert result["is_superadditive"] is True, "Expected is_superadditive=True"
    print(f"✓ TEST 4 PASSED: Superadditivity index = {result['superadditivity_index']:.3f}")
    return result


# ─────────────────────────────────────────────
# TEST 5: Superadditivity — NOT superadditive
# ─────────────────────────────────────────────
def test_superadditivity_negative() -> dict:
    episodes_2v = [
        {"first_contact_step_any": 87, "seed": 0, "num_villains": 2, "outcome": "hero_escaped"},
    ]
    episodes_1v = [
        {"first_contact_step_any": 87, "seed": 0, "num_villains": 1, "outcome": "hero_escaped"},
    ]
    result = compute_superadditivity(episodes_2v, episodes_1v)
    assert result["superadditivity_index"] is not None and result["superadditivity_index"] <= 1.0, (
        f"Expected index <= 1.0, got {result['superadditivity_index']}"
    )
    print(f"✓ TEST 5 PASSED: Non-superadditive correctly identified = {result['superadditivity_index']:.3f}")
    return result


# ─────────────────────────────────────────────
# TEST 6: Phase transition — SHARP transition
# ─────────────────────────────────────────────
def _mv_from_bin(b: int) -> list[float]:
    """Unit vector in bin b (0..7); small jitter so length is non-zero."""
    ang = (b * 2 * math.pi / 8.0) + 0.02
    return [math.cos(ang), math.sin(ang)]


def test_phase_transition_sharp() -> dict:
    """
    Phase A (0..59): fixed (v1 bin 0, v2 bin 2) every step → joint mass on one cell → MI ≈ 0.
    Phase B (60..99): v2 bin = (v1_bin + 1) % 8 with v1 cycling → coupling → MI ≫ 0.
    """
    steps: list = []
    for i in range(60):
        steps.append(
            {
                "step_index": i,
                "hero_position": [80.0, 80.0, 0.0],
                "villain_positions": {"villain_1": [50.0, 50.0, 0.0], "villain_2": [30.0, 30.0, 0.0]},
                "per_agent": [
                    {"agent_id": "hero_1", "role": "hero", "actual_movement": [0.5, 0.5]},
                    {"agent_id": "villain_1", "role": "villain", "actual_movement": _mv_from_bin(0)},
                    {"agent_id": "villain_2", "role": "villain", "actual_movement": _mv_from_bin(2)},
                ],
            }
        )
    for i in range(40):
        j = i + 60
        b1 = i % 8
        b2 = (b1 + 1) % 8
        steps.append(
            {
                "step_index": j,
                "hero_position": [80.0, 80.0, 0.0],
                "villain_positions": {"villain_1": [50.0, 50.0, 0.0], "villain_2": [30.0, 30.0, 0.0]},
                "per_agent": [
                    {"agent_id": "hero_1", "role": "hero", "actual_movement": [0.5, 0.5]},
                    {"agent_id": "villain_1", "role": "villain", "actual_movement": _mv_from_bin(b1)},
                    {"agent_id": "villain_2", "role": "villain", "actual_movement": _mv_from_bin(b2)},
                ],
            }
        )
    result = detect_phase_transition(steps, window_size=10, mi_threshold=0.5)
    assert result["transition_detected"] is True, f"Expected transition detected, got {result}"
    ts = result["transition_step"]
    # First sustained high-MI windows can start mid-50s once the sliding window mixes into phase B.
    assert ts is not None and 50 <= ts <= 65, f"Expected transition near step ~60, got {ts}"
    print(f"✓ TEST 6 PASSED: Phase transition at step {ts}, sharpness={result['sharpness']:.2f}")
    return result


# ─────────────────────────────────────────────
# TEST 7: Information threshold extraction
# ─────────────────────────────────────────────
def test_information_threshold() -> dict:
    steps = []
    for i in range(50):
        steps_blind = i if i >= 5 else 0
        tgt = (98.0, 98.0) if i >= 25 else (40.0, 40.0)
        steps.append(
            _synth_step(
                i,
                v2_target=tgt,
                v1_sees_hero=True,
                v2_blind=True,
                v2_steps_since=steps_blind,
            )
        )
    beacon_result = detect_beacon_behavior(steps)
    threshold_result = analyze_information_threshold(steps, beacon_result)
    if beacon_result["beacon_detected"]:
        assert threshold_result["threshold_at_beacon_onset"] is not None, "Expected threshold value"
        t0 = threshold_result["threshold_at_beacon_onset"]
        assert 15 <= t0 <= 30, f"Expected threshold in plausible blind range, got {t0}"
    print(f"✓ TEST 7 PASSED: Threshold = {threshold_result.get('threshold_at_beacon_onset')} steps blind")
    return threshold_result


# ─────────────────────────────────────────────
# TEST 8: Edge cases
# ─────────────────────────────────────────────
def test_edge_cases() -> None:
    result = detect_beacon_behavior([])
    assert result["beacon_detected"] is False

    short_steps = []
    for i in range(3):
        short_steps.append(
            _synth_step(
                i,
                v2_target=(98.0, 98.0),
                v1_sees_hero=True,
                v2_blind=True,
                v2_steps_since=10,
            )
        )
    result = detect_beacon_behavior(short_steps)
    assert result["beacon_detected"] is False, "Short episode should not trigger beacon (below min_duration)"
    print("✓ TEST 8 PASSED: Edge cases handled correctly")


# ─────────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("PHASE 1 METRIC VERIFICATION TESTS")
    print("=" * 50 + "\n")

    failed: list[str] = []

    tests = [
        ("Beacon Detected", test_beacon_detected),
        ("Beacon Not Detected", test_beacon_not_detected),
        ("Beacon V1 Blind (ToM)", test_beacon_v1_blind),
        ("Superadditivity Positive", test_superadditivity_positive),
        ("Superadditivity Negative", test_superadditivity_negative),
        ("Phase Transition Sharp", test_phase_transition_sharp),
        ("Information Threshold", test_information_threshold),
        ("Edge Cases", test_edge_cases),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
        except AssertionError as e:
            print(f"✗ FAILED: {name}")
            print(f"  Error: {e}")
            failed.append(name)
        except Exception as e:
            print(f"✗ ERROR: {name}")
            print(f"  Exception: {e}")
            import traceback

            traceback.print_exc()
            failed.append(name)

    print("\n" + "=" * 50)
    if not failed:
        print("ALL TESTS PASSED — Ready to run Phase 1 episodes")
        print("=" * 50)
        print("\nNext step:")
        print("Run the Phase 1 dry-run commands from the verification checklist.")
        sys.exit(0)
    else:
        print(f"FAILED TESTS ({len(failed)}/{len(tests)}):")
        for f in failed:
            print(f"  - {f}")
        print("=" * 50)
        print("\nFix failing metrics before running episodes.")
        sys.exit(1)
