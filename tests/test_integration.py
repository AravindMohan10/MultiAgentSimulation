"""
Integration tests: engine + run_episode (rule-based, no API keys).

Run from repo root:
  PYTHONPATH=. python tests/test_integration.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.engine import SimulationEngine
from src.core.models import AgentConfig, AgentType, EnvironmentConfig, MapTemplate
from src.experiments.runner import EpisodeConfig, run_episode
from src.metrics import (
    analyze_information_threshold,
    compute_role_separation,
    compute_superadditivity,
    detect_beacon_behavior,
    detect_phase_transition,
    effective_v2_detection_radius_at_contact,
    role_divergence,
)


def _minimal_rule_based_episode(*, max_steps: int = 30) -> EpisodeConfig:
    env = EnvironmentConfig(
        world_size=(80.0, 80.0),
        max_steps=max_steps,
        seed=42,
        map_template=MapTemplate.SCATTERED,
        obstacle_density=0.06,
        num_villains=2,
        villain_hero_sight_radius=20.0,
        visibility_radius=50.0,
    )
    agents = [
        AgentConfig(
            id="hero_1",
            agent_type=AgentType.HERO,
            strategy_mode="rule_based",
            max_speed=1.2,
            vision_radius=40.0,
            use_auto_coord_message=False,
        ),
        AgentConfig(
            id="villain_1",
            agent_type=AgentType.VILLAIN,
            strategy_mode="rule_based",
            max_speed=1.0,
            vision_radius=40.0,
            use_auto_coord_message=False,
        ),
        AgentConfig(
            id="villain_2",
            agent_type=AgentType.VILLAIN,
            strategy_mode="rule_based",
            max_speed=1.0,
            vision_radius=40.0,
            use_auto_coord_message=False,
        ),
    ]
    return EpisodeConfig(
        episode_id="integration_rule_based",
        environment=env,
        agent_configs=agents,
        capture_radius=2.0,
    )


def test_simulation_engine_reset_and_observations() -> None:
    cfg = _minimal_rule_based_episode().environment
    agent_cfgs = _minimal_rule_based_episode().agent_configs
    engine = SimulationEngine(cfg, agent_cfgs)
    ws = engine.reset()
    assert ws.step_index == 0
    assert "hero_1" in ws.agents
    assert sum(1 for a in ws.agents.values() if a.agent_type == AgentType.VILLAIN) == 2
    obs = engine.get_observations()
    assert len(obs) == 3
    for aid in ("hero_1", "villain_1", "villain_2"):
        assert aid in obs


def test_run_episode_rule_based_terminates() -> None:
    ep = _minimal_rule_based_episode(max_steps=40)
    out = run_episode(ep, {}, log_dir=None)
    assert out.result in ("hero_captured", "hero_escaped")
    assert out.steps >= 1
    assert out.episode_id == "integration_rule_based"


def test_run_episode_writes_log_artifacts() -> None:
    ep = _minimal_rule_based_episode(max_steps=15)
    with tempfile.TemporaryDirectory() as td:
        log_dir = Path(td)
        out = run_episode(ep, {}, log_dir=log_dir)
        assert (log_dir / f"{ep.episode_id}_summary.json").is_file()
        assert (log_dir / f"{ep.episode_id}_steps.jsonl").is_file()
        assert (log_dir / f"{ep.episode_id}_config.json").is_file()
        summary = json.loads((log_dir / f"{ep.episode_id}_summary.json").read_text(encoding="utf-8"))
        assert summary.get("outcome") == out.result


def test_metrics_package_exports_for_batch_analysis() -> None:
    """Smoke-test the same imports ``scripts/analyze_batch.py`` uses."""
    assert callable(compute_superadditivity)
    assert callable(analyze_information_threshold)
    assert callable(detect_phase_transition)
    assert callable(detect_beacon_behavior)
    assert callable(compute_role_separation)
    assert callable(effective_v2_detection_radius_at_contact)
    assert callable(role_divergence)


def test_role_separation_on_synthetic_step() -> None:
    steps = [
        {
            "step_index": 0,
            "hero_position": [40.0, 40.0, 0.0],
            "villain_positions": {"villain_1": [30.0, 30.0, 0.0], "villain_2": [50.0, 50.0, 0.0]},
            "per_agent": [
                {
                    "agent_id": "hero_1",
                    "actual_position": [40.0, 40.0],
                    "intent": "flee_threat",
                    "movement": [0.1, 0.0, 0.0],
                },
                {
                    "agent_id": "villain_1",
                    "actual_position": [30.0, 30.0],
                    "intent": "pursue_target",
                    "movement": [1.0, 0.0, 0.0],
                },
                {
                    "agent_id": "villain_2",
                    "actual_position": [50.0, 50.0],
                    "intent": "cut_off",
                    "movement": [0.0, 1.0, 0.0],
                },
            ],
        }
    ]
    r = compute_role_separation(steps, window=1)
    assert "divergence_score" in r
    assert "spontaneous_divergence_fraction" in r


def main() -> None:
    tests = [
        ("engine reset + observations", test_simulation_engine_reset_and_observations),
        ("run_episode rule-based", test_run_episode_rule_based_terminates),
        ("run_episode log artifacts", test_run_episode_writes_log_artifacts),
        ("metrics exports (batch analysis)", test_metrics_package_exports_for_batch_analysis),
        ("role_separation synthetic", test_role_separation_on_synthetic_step),
    ]
    failed: list[str] = []
    for name, fn in tests:
        try:
            fn()
            print(f"✓ {name}")
        except AssertionError as e:
            print(f"✗ {name}: {e}")
            failed.append(name)
        except Exception as e:
            print(f"✗ {name}: {e!r}")
            failed.append(name)
            raise
    if failed:
        print(f"\nFailed: {failed}")
        sys.exit(1)
    print("\nAll integration tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
