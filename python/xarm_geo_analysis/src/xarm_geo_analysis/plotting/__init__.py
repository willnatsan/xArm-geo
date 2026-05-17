from xarm_geo_analysis.plotting.style import apply_style
from xarm_geo_analysis.plotting.tracking import (
    plot_3d_path,
    plot_3d_paths_compare,
    plot_tracking_errors,
    plot_settling,
    plot_error_twist_overlay,
    plot_axis_zoom,
    plot_error_boxplot,
)
from xarm_geo_analysis.plotting.torque import plot_torque_triplet, plot_velocity_triplet
from xarm_geo_analysis.plotting.compare import overlay
from xarm_geo_analysis.plotting.disturbance import (
    plot_tracking_with_disturbance,
    plot_integrator_state,
)
from xarm_geo_analysis.plotting.safety import (
    plot_obstacle_distance,
    plot_intervention_norm,
    plot_command_norm_overlay,
)

apply_style()

__all__ = [
    # single-trial
    "plot_3d_path",
    "plot_tracking_errors",
    "plot_settling",
    "plot_torque_triplet",
    "plot_velocity_triplet",
    "plot_axis_zoom",
    "plot_integrator_state",
    "plot_intervention_norm",
    # multi-trial / experiment
    "overlay",
    "plot_3d_paths_compare",
    "plot_error_twist_overlay",
    "plot_error_boxplot",
    "plot_tracking_with_disturbance",
    "plot_obstacle_distance",
    "plot_command_norm_overlay",
]
