#pragma once

#include <string>
#include <vector>

#include <GLFW/glfw3.h>
#include <mujoco/mjdata.h>
#include <mujoco/mujoco.h>

#include <xarm_geo/core/manifold.h>
#include <xarm_geo/interfaces/interface.h>

namespace xarm_geo {
    class Simulation {
    public:
        // --- MuJoCo Configuration Struct ---

        struct Config {
            int window_width = 1200;
            int window_height = 900;
            std::string window_title = "xArm-Geo Simulator";
            bool vsync_enabled = true;

            double camera_distance = -1.0;  // -1 = Use MuJoCo Default
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

        // --- Direct Access (If Needed) ---

        mjModel *model = nullptr;
        mjData *data = nullptr;

        // --- Lifecycle Management ---

        explicit Simulation(const std::string &mjcf_file);
        Simulation(const std::string &mjcf_file, const Config &config);
        ~Simulation();

        auto initialised() -> bool;
        void shutdown();
        void reset() const;
        [[nodiscard]] auto is_running() const -> bool;

        // --- Concept: Observable (READ) ---

        void read(JointPosition &pos) const;
        void read(JointVelocity &vel) const;
        void read(JointTorque &tau) const;

        // --- Concept: Controllable (WRITE) ---

        // void write(const JointPosition &pos);
        void write(const JointVelocity &vel);
        // void write(const JointTorque &tau);

        // --- Simulation Stepping ---

        void step();
        void step(int n_steps);

        // --- Simulation Queries ---

        [[nodiscard]] auto get_timestep() const -> double { return model->opt.timestep; }
        [[nodiscard]] auto get_time() const -> double { return data->time; }
        [[nodiscard]] auto get_dof() const -> int { return dof_; }

        [[nodiscard]] auto get_pose(const std::string &body_name = "") const -> manifold::SE3;
        [[nodiscard]] auto get_pose(int body_id) const -> manifold::SE3;
        [[nodiscard]] auto get_pose_tree() const -> std::vector<manifold::SE3>;
        [[nodiscard]] auto get_jacobian(const std::string &body_name) const
            -> manifold::SE3::Jacobian;
        [[nodiscard]] auto get_twist(const std::string &body_name,
                                     const Eigen::VectorXd &q_dot) const -> manifold::SE3::Twist;

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

        GLFWwindow *window_ = nullptr;
        mjvCamera camera_{};
        mjvOption option_{};
        mjvScene scene_{};
        mjrContext context_{};

        [[nodiscard]] auto get_body_id(const std::string &body_name) const -> int;

        // --- GLFW Mouse Callbacks ---

        static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
        static void mouse_move_callback(GLFWwindow *window, double x_pos, double y_pos);
        static void scroll_callback(GLFWwindow *window, double x_offset, double y_offset);
    };
}  // namespace xarm_geo
