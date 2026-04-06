"""LLMActionOutput accepts message as list or {payload, channel}."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agents.schema import LLMActionOutput


def test_message_shorthand_list() -> None:
    m = LLMActionOutput.model_validate(
        {
            "intent": "pursue_target",
            "target_position": [50.0, 50.0],
            "message": [0.0, 0.0, 0.0, 10.0, 20.0],
        }
    )
    assert m.message is not None
    assert m.message.payload[:5] == [0.0, 0.0, 0.0, 10.0, 20.0]
    assert m.message.channel == "coord"


def test_message_object_form() -> None:
    m = LLMActionOutput.model_validate(
        {
            "intent": "pursue_target",
            "target_position": [1.0, 2.0],
            "message": {"payload": [1.0, 2.0, 1.0, 3.0, 4.0], "channel": "coord"},
        }
    )
    assert m.message is not None
    assert len(m.message.payload) >= 5


if __name__ == "__main__":
    test_message_shorthand_list()
    test_message_object_form()
    print("ok")
