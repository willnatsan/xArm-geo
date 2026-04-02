import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Load the data
df = pd.read_csv("tests/results/trajectory_log_2.csv")

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

# 1. Plot the continuous actual trajectory line (faint gray so the frames stand out)
ax.plot(
    df["target_x"],
    df["target_y"],
    df["target_z"],
    color="gray",
    alpha=0.5,
    linewidth=2,
    label="Trajectory Path",
)

# 2. DOWNSAMPLE the data for the coordinate frames
step_size = 100  # Draw a frame every 100 simulation steps
df_sampled = df.iloc[::step_size]

axis_length = 0.02  # Length of the arrows in meters

# 3. Loop through the sampled data and draw the triads
for index, row in df_sampled.iterrows():
    pos = [row["actual_x"], row["actual_y"], row["actual_z"]]

    # Eigen's eulerAngles(2, 1, 0) maps to intrinsic ZYX in SciPy
    yaw = row["actual_yaw"]
    pitch = row["actual_pitch"]
    roll = row["actual_roll"]

    # Convert Euler angles to a 3x3 Rotation Matrix
    rot = R.from_euler("ZYX", [yaw, pitch, roll], degrees=False)
    rot_matrix = rot.as_matrix()

    # The columns of the rotation matrix represent the local X, Y, Z axes in world coordinates
    x_axis = rot_matrix[:, 0]
    y_axis = rot_matrix[:, 1]
    z_axis = rot_matrix[:, 2]

    # Plot Local X axis (Red)
    ax.quiver(
        *pos,
        *x_axis,
        color="r",
        length=axis_length,
        normalize=True,
        arrow_length_ratio=0.3,
    )
    # Plot Local Y axis (Green)
    ax.quiver(
        *pos,
        *y_axis,
        color="g",
        length=axis_length,
        normalize=True,
        arrow_length_ratio=0.3,
    )
    # Plot Local Z axis (Blue)
    ax.quiver(
        *pos,
        *z_axis,
        color="b",
        length=axis_length,
        normalize=True,
        arrow_length_ratio=0.3,
    )

# Add dummy legend entries for the RGB axes
ax.plot([], [], color="r", label="Local X")
ax.plot([], [], color="g", label="Local Y")
ax.plot([], [], color="b", label="Local Z")

# Formatting
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.set_title("End-Effector Pose (Coordinate Frames along Trajectory)")
ax.legend()

# Equal aspect ratio trick to prevent the 3D view from stretching the shape
limits = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
ranges = [abs(lim[1] - lim[0]) for lim in limits]
max_range = max(ranges) / 2.0
mid_x = sum(limits[0]) / 2.0
mid_y = sum(limits[1]) / 2.0
mid_z = sum(limits[2]) / 2.0
ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
ax.set_zlim3d([mid_z - max_range, mid_z + max_range])

plt.show()
