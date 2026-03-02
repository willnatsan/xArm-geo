import os
import xacrodoc

"""
NOTE:
The generated URDFs & MJCF XMLs will likely NOT have the right Inertial Params!
This is because they're not being parsed with the `robot_sn` (Robot's Serial Number) parameter.
The main `Model` class contains the correct Kinematic & Inertial Params; That's what the API's calculations use.
"""

ASSETS_DIR = os.path.dirname(__file__)
MESH_DIR = os.path.join(ASSETS_DIR, "meshes")
URDF_SRC_DIR = os.path.join(ASSETS_DIR, "urdf")
URDF_COMPILED_DIR = os.path.join(ASSETS_DIR, "urdf/_compiled")
MJCF_DIR = os.path.join(ASSETS_DIR, "mjcf")

xacrodoc.packages.update_package_cache({"xarm_description": str(ASSETS_DIR)})


def gen_urdf(urdf_filename, dof, robot_type):
    xacro_file = os.path.join(URDF_SRC_DIR, "xarm_device.urdf.xacro")
    urdf_file = os.path.join(URDF_COMPILED_DIR, urdf_filename + ".urdf")

    doc = xacrodoc.XacroDoc.from_file(
        xacro_file, subargs={"dof": str(dof), "robot_type": robot_type}
    )
    doc.to_urdf_file(urdf_file)


def gen_mjcf(mjcf_filename, dof, robot_type):
    xacro_file = os.path.join(URDF_SRC_DIR, "xarm_device.urdf.xacro")
    mjcf_file = os.path.join(MJCF_DIR, mjcf_filename + ".xml")

    doc = xacrodoc.XacroDoc.from_file(
        xacro_file, subargs={"dof": str(dof), "robot_type": robot_type}
    )
    doc.to_mjcf_file(
        mjcf_file,
        strippath="false",
        discardvisual="false",
        fusestatic="false",
        autolimits="true",
    )


def main():
    args = [
        {"dof": 5, "robot_type": "xarm", "filename": "xarm5"},
        {"dof": 6, "robot_type": "xarm", "filename": "xarm6"},
        {"dof": 7, "robot_type": "xarm", "filename": "xarm7"},
        {"dof": 6, "robot_type": "lite", "filename": "lite6"},
        {"dof": 6, "robot_type": "uf850", "filename": "uf850"},
        {"dof": 7, "robot_type": "xarm7_mirror", "filename": "xarm7_mirror"},
    ]

    for arg in args:
        gen_urdf(arg["filename"], arg["dof"], arg["robot_type"])
        gen_mjcf(arg["filename"], arg["dof"], arg["robot_type"])


if __name__ == "__main__":
    main()
