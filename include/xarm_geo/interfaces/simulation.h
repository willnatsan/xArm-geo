#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>
#include <mujoco/mjdata.h>
#include <mujoco/mujoco.h>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/interfaces/interface.h>

namespace xarm_geo {

    enum class ControlMode : std::uint8_t { VELOCITY, TORQUE };

    class Simulation {

    public:
        // --- MuJoCo Configuration ---

        struct Config {
            int window_width = 1200;
            int window_height = 900;
            std::string window_title = "xArm-Geo Simulator";
            bool vsync_enabled = true;

            double camera_distance = -1.0;  // -1 -> MuJoCo default
            double camera_azimuth = 0.0;
            double camera_elevation = -45.0;
            Eigen::Vector3d camera_lookat{0, 0, 0};
            bool camera_use_defaults = true;

            bool enable_textures = true;
            bool enable_lighting = true;
            bool show_decorations = true;
            int font_scale = mjFONTSCALE_150;

            int max_scene_geoms = 2000;
            std::string marker_name = "marker";
            double scroll_sensitivity = 0.05;
        };

        // --- Direct MuJoCo Access ---

        mjModel *model = nullptr;
        mjData *data = nullptr;

        // --- Constructors & Destructor ---

        explicit Simulation(const std::string &mjcf_file);
        Simulation(const std::string &mjcf_file, const Config &config);
        ~Simulation();

        // --- Interface Concept: Lifecycle & State Reading ---

        [[nodiscard]] auto is_running() const -> bool;
        void shutdown();
        auto read(JointState &state) noexcept -> InterfaceStatus;
        [[nodiscard]] auto read_time() const noexcept -> std::chrono::nanoseconds {
            return last_read_time_;
        }

        // --- Controllable Concept: Command Writing ---

        auto write(const JointVelocity &vel) noexcept -> InterfaceStatus;
        auto write(const JointTorque &torque) noexcept -> InterfaceStatus;

        // --- Simulation Stepping ---

        void step();
        void step(int n_steps);

        // --- Simulation Getters ---

        [[nodiscard]] auto get_timestep() const -> double { return model->opt.timestep; }
        [[nodiscard]] auto get_time() const -> double { return data->time; }
        [[nodiscard]] auto get_dof() const -> int { return dof_; }

        [[nodiscard]] auto get_body_id(const std::string &body_name) const -> int;
        [[nodiscard]] auto get_pose(const std::string &body_name = "") const -> manifold::SE3;
        [[nodiscard]] auto get_pose(int body_id) const -> manifold::SE3;
        [[nodiscard]] auto get_pose_tree() const -> std::vector<manifold::SE3>;
        [[nodiscard]] auto get_jacobian(const std::string &body_name) const
            -> manifold::SE3::Jacobian;
        [[nodiscard]] auto get_twist(const std::string &body_name, const Eigen::VectorXd &v) const
            -> manifold::SE3::Twist;

        // --- Simulation Setters ---

        void set_joint_positions(const Eigen::VectorXd &q) const;
        void set_control_mode(ControlMode mode);
        void reset() const;

        // --- Rendering & Visualisation ---

        void render();
        void update_scene();
        void update_model() const;
        void set_marker(const manifold::SE3 &pose) const;
        void draw_arrow(const Eigen::Vector3d &point_origin, const Eigen::Vector3d &point_end,
                        double width, const float rgba[4]);

    private:
        Config config_;
        int dof_;
        int marker_id_ = -1;
        bool is_shutdown_ = false;
        std::chrono::nanoseconds last_read_time_{0};
        ControlMode current_mode_ = ControlMode::VELOCITY;

        // Cached actuator parameters; reused by set_control_mode().
        std::vector<double> kv_gains_;
        std::vector<double> force_limits_;
        std::vector<double> vel_limits_;

        // MuJoCo GUI.
        GLFWwindow *window_ = nullptr;
        mjvCamera camera_{};
        mjvOption option_{};
        mjvScene scene_{};
        mjrContext context_{};

        // --- Static GLFW Mouse Callbacks ---

        static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
        static void mouse_move_callback(GLFWwindow *window, double x_pos, double y_pos);
        static void scroll_callback(GLFWwindow *window, double x_offset, double y_offset);
    };

    // --- Compile-Time Concept Verifications ---

    static_assert(Interface<Simulation>);
    static_assert(Controllable<Simulation, JointVelocity>);
    static_assert(Controllable<Simulation, JointTorque>);

}  // namespace xarm_geo
