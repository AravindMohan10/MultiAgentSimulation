from .capture import capture_rate, capture_time
from .communication import message_utilization_score
from .coordination import redundancy_score
from .efficiency import path_efficiency, role_divergence as cross_episode_role_divergence
from .role_divergence_metrics import within_episode_role_divergence
from .beacon_behavior import detect_beacon_behavior
from .role_separation import compute_role_separation
from .stuck import stuck_rate_per_agent
from .map_utilization import chokepoint_proximity_score, spoke_coverage_score
from .superadditivity import compute_superadditivity, effective_v2_detection_radius_at_contact
from .phase_transition import detect_phase_transition, mi_for_window
from .information_threshold import analyze_information_threshold

# ``role_divergence`` is the entropy function from efficiency.py (not the metrics module).
role_divergence = cross_episode_role_divergence

__all__ = [
    "capture_rate",
    "capture_time",
    "message_utilization_score",
    "redundancy_score",
    "path_efficiency",
    "role_divergence",
    "within_episode_role_divergence",
    "detect_beacon_behavior",
    "compute_role_separation",
    "compute_superadditivity",
    "effective_v2_detection_radius_at_contact",
    "detect_phase_transition",
    "mi_for_window",
    "analyze_information_threshold",
    "stuck_rate_per_agent",
    "chokepoint_proximity_score",
    "spoke_coverage_score",
]
