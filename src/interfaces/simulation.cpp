#include <mujoco/mjmodel.h>
#include <stdexcept>

#include <xarm_geo/interfaces/simulation.h>
#include <xarm_geo_config.h>

namespace xarm_geo {

    // --- Static State For Mouse Callbacks ---

    static bool g_button_left = false;
    static bool g_button_middle = false;
    static bool g_button_right = false;
    static double g_last_x = 0;
    static double g_last_y = 0;

    // --- Constructors & Destructors ---

    Simulation::Simulation(const std::string &mjcf_file) : Simulation(mjcf_file, Config{}) {}

    Simulation::Simulation(const std::string &mjcf_file, const Config &config) : config_(config) {
        char error[1000];
        this->model = mj_loadXML((MJCF_PATH + mjcf_file).c_str(), nullptr, error, 1000);
        if (!this->model) {
            throw std::runtime_error("MuJoCo Model Load Error: " + std::string(error));
        }
        this->dof_ = this->model->nv;
        if (this->dof_ == 0) {
            throw std::runtime_error("MuJoCo Model Load Error: Model has 0 DOF");
        }
        this->data = mj_makeData(this->model);

        // Cache actuator parameters from the parsed XML.
        this->kv_gains_.resize(this->model->nu);
        this->force_limits_.resize(this->model->nu);
        this->vel_limits_.resize(this->model->nu);

        for (int i = 0; i < this->model->nu; ++i) {
            this->kv_gains_[i] = this->model->actuator_gainprm[(i * mjNGAIN) + 0];
            this->vel_limits_[i] = this->model->actuator_ctrlrange[(i * 2) + 1];
            this->force_limits_[i] = this->model->actuator_forcerange[(i * 2) + 1];
        }

        this->current_mode_ = ControlMode::VELOCITY;

        // Marker disabled if the configured body is not in the model.
        try {
            this->marker_id_ = this->model->body_mocapid[this->get_body_id(config.marker_name)];
        } catch (const std::runtime_error &) { this->marker_id_ = -1; }

        // --- GLFW Window & Visualisation ---
        if (!glfwInit()) {
            throw std::runtime_error("Visualisation Error: Failed to initialize GLFW");
        }
        this->window_ = glfwCreateWindow(config.window_width, config.window_height,
                                         config.window_title.c_str(), nullptr, nullptr);
        if (!this->window_) {
            throw std::runtime_error("Visualisation Error: Failed to create GLFW window");
        }

        glfwMakeContextCurrent(this->window_);
        glfwSwapInterval(config.vsync_enabled ? 1 : 0);
        glfwSetWindowUserPointer(this->window_, this);

        mjv_defaultOption(&this->option_);
        this->option_.flags[mjVIS_TEXTURE] = config.enable_textures ? 1 : 0;
        this->option_.flags[mjVIS_LIGHT] = config.enable_lighting ? 1 : 0;
        this->option_.geomgroup[mjCAT_DECOR] = config.show_decorations ? 1 : 0;

        mjv_defaultCamera(&this->camera_);
        if (!config.camera_use_defaults) {
            this->camera_.distance = config.camera_distance;
            this->camera_.azimuth = config.camera_azimuth;
            this->camera_.elevation = config.camera_elevation;
            this->camera_.lookat[0] = config.camera_lookat.x();
            this->camera_.lookat[1] = config.camera_lookat.y();
            this->camera_.lookat[2] = config.camera_lookat.z();
        }

        mjr_defaultContext(&this->context_);
        mjv_makeScene(this->model, &this->scene_, config.max_scene_geoms);
        mjr_makeContext(this->model, &this->context_, config.font_scale);

        // Callbacks: Backspace resets the simulation; mouse handles camera orbit/zoom.
        glfwSetKeyCallback(
            this->window_, [](GLFWwindow *window, int key, int scancode, int act, int mods) {
                auto *sim = static_cast<Simulation *>(glfwGetWindowUserPointer(window));
                if (!sim) return;
                if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE) {
                    mj_resetData(sim->model, sim->data);
                    mj_forward(sim->model, sim->data);
                }
            });

        glfwSetMouseButtonCallback(this->window_, mouse_button_callback);
        glfwSetCursorPosCallback(this->window_, mouse_move_callback);
        glfwSetScrollCallback(this->window_, scroll_callback);
    }

    Simulation::~Simulation() { this->shutdown(); }

    // --- Interface Concept: Lifecycle & State Reading ---

    auto Simulation::is_running() const -> bool {
        if (this->is_shutdown_) return false;

        // UI close button dictates lifecycle.
        if (this->window_) { return !glfwWindowShouldClose(this->window_); }

        return true;
    }

    void Simulation::shutdown() {
        if (this->is_shutdown_) return;

        if (this->window_) {
            glfwDestroyWindow(this->window_);
            this->window_ = nullptr;
        }
        glfwTerminate();

        mjv_freeScene(&this->scene_);
        mjr_freeContext(&this->context_);

        if (this->data) {
            mj_deleteData(this->data);
            this->data = nullptr;
        }
        if (this->model) {
            mj_deleteModel(this->model);
            this->model = nullptr;
        }

        is_shutdown_ = true;
    }

    auto Simulation::read(JointState &state) noexcept -> InterfaceStatus {
        if (this->is_shutdown_ || !this->data) { return InterfaceStatus::ERROR; };

        if (state.q.size() != this->dof_) { state.q.resize(this->dof_); }
        if (state.v.size() != this->dof_) { state.v.resize(this->dof_); }
        if (state.tau.size() != this->dof_) { state.tau.resize(this->dof_); }

        state.q = Eigen::Map<const Eigen::VectorXd>(this->data->qpos, this->dof_);
        state.v = Eigen::Map<const Eigen::VectorXd>(this->data->qvel, this->dof_);
        state.tau = Eigen::Map<const Eigen::VectorXd>(this->data->qfrc_actuator, this->dof_);

        this->last_read_time_ =
            std::chrono::nanoseconds(static_cast<long long>(this->data->time * 1e9));

        return InterfaceStatus::OK;
    }

    // --- Controllable Concept: Command Writing ---

    auto Simulation::write(const JointVelocity &vel) noexcept -> InterfaceStatus {
        if (this->current_mode_ != ControlMode::VELOCITY) { return InterfaceStatus::ERROR; }
        if (this->is_shutdown_ || vel.v.size() != this->dof_) { return InterfaceStatus::ERROR; };

        Eigen::Map<Eigen::VectorXd>(this->data->ctrl, this->model->nu) = vel.v;

        return InterfaceStatus::OK;
    }

    auto Simulation::write(const JointTorque &torque) noexcept -> InterfaceStatus {
        if (this->current_mode_ != ControlMode::TORQUE) { return InterfaceStatus::ERROR; }
        if (this->is_shutdown_ || torque.tau.size() != this->dof_) {
            return InterfaceStatus::ERROR;
        };

        Eigen::Map<Eigen::VectorXd>(this->data->ctrl, this->model->nu) = torque.tau;

        return InterfaceStatus::OK;
    }

    // --- Simulation Stepping ---

    void Simulation::step() { mj_step(this->model, this->data); }

    void Simulation::step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) mj_step(this->model, this->data);
    }

    // --- Simulation Getters ---

    auto Simulation::get_body_id(const std::string &body_name) const -> int {
        int id = mj_name2id(this->model, mjOBJ_BODY, body_name.c_str());
        if (id == -1) { throw std::runtime_error("Error: Could not find body: " + body_name); }
        return id;
    }

    auto Simulation::get_pose(const std::string &body_name) const -> manifold::SE3 {
        int body_id = body_name.empty() ? (this->model->nbody - 1) : get_body_id(body_name);
        return get_pose(body_id);
    }

    auto Simulation::get_pose(const int body_id) const -> manifold::SE3 {
        const Eigen::Vector3d pos(this->data->xpos + (body_id * 3));

        // Read MuJoCo's quaternion (w, x, y, z) directly rather than converting from xmat.
        const mjtNum *q = this->data->xquat + (body_id * 4);
        manifold::SO3 rot(Eigen::Quaterniond(q[0], q[1], q[2], q[3]));

        return {rot, pos};
    }

    auto Simulation::get_pose_tree() const -> std::vector<manifold::SE3> {
        std::vector<manifold::SE3> poses;
        poses.reserve(this->model->nbody - 1);
        for (int i = 1; i < this->model->nbody; i++) { poses.push_back(this->get_pose(i)); }
        return poses;
    }

    auto Simulation::get_jacobian(const std::string &body_name) const -> manifold::SE3::Jacobian {
        const int body_id = this->get_body_id(body_name);
        manifold::SE3::Jacobian J_b(6, this->dof_);

        Eigen::Matrix<mjtNum, 3, Eigen::Dynamic, Eigen::RowMajor> J_pos(3, this->model->nv);
        Eigen::Matrix<mjtNum, 3, Eigen::Dynamic, Eigen::RowMajor> J_rot(3, this->model->nv);

        // Space-frame Jacobian.
        mj_jac(this->model, this->data, J_pos.data(), J_rot.data(), this->data->xpos, body_id);

        J_b.topRows(3) = J_pos.leftCols(this->dof_);
        J_b.bottomRows(3) = J_rot.leftCols(this->dof_);

        return J_b;
    }

    auto Simulation::get_twist(const std::string &body_name, const Eigen::VectorXd &v) const
        -> manifold::SE3::Twist {

        if (v.size() != this->dof_) {
            throw std::invalid_argument("get_twist: Vector size does not match DOF");
        }

        return this->get_jacobian(body_name) * v;
    }

    // --- Simulation Setters ---

    void Simulation::set_joint_positions(const Eigen::VectorXd &q) const {
        if (q.size() != this->dof_) {
            throw std::invalid_argument("set_joint_positions: Vector size does not match DOF");
        }

        for (int i = 0; i < this->dof_; ++i) { data->qpos[i] = q[i]; }
        mj_forward(model, data);
    }

    void Simulation::set_control_mode(ControlMode mode) {
        if (mode == current_mode_) return;

        for (int i = 0; i < this->model->nu; ++i) {
            if (mode == ControlMode::VELOCITY) {
                // Velocity mode: F = kv * ctrl - kv * vel (affine bias).
                this->model->actuator_biastype[i] = mjBIAS_AFFINE;
                this->model->actuator_gainprm[(i * mjNGAIN) + 0] = this->kv_gains_[i];
                this->model->actuator_biasprm[(i * mjNBIAS) + 1] = 0.0;
                this->model->actuator_biasprm[(i * mjNBIAS) + 2] = -this->kv_gains_[i];

                this->model->actuator_ctrlrange[(i * 2) + 0] = -this->vel_limits_[i];
                this->model->actuator_ctrlrange[(i * 2) + 1] = this->vel_limits_[i];

            } else if (mode == ControlMode::TORQUE) {
                // Torque mode: F = ctrl (no bias).
                this->model->actuator_biastype[i] = mjBIAS_NONE;
                this->model->actuator_gainprm[(i * mjNGAIN) + 0] = 1.0;
                this->model->actuator_biasprm[(i * mjNBIAS) + 1] = 0.0;
                this->model->actuator_biasprm[(i * mjNBIAS) + 2] = 0.0;

                this->model->actuator_ctrlrange[(i * 2) + 0] = -this->force_limits_[i];
                this->model->actuator_ctrlrange[(i * 2) + 1] = this->force_limits_[i];
            }
        }

        // Zero the control buffer to prevent jumps during the transition.
        Eigen::Map<Eigen::VectorXd>(this->data->ctrl, this->model->nu).setZero();

        current_mode_ = mode;
    }

    void Simulation::reset() const {
        Eigen::Map<Eigen::VectorXd>(this->data->qvel, this->dof_).setZero();
        Eigen::Map<Eigen::VectorXd>(this->data->ctrl, this->model->nu).setZero();
    }

    void Simulation::apply_external_wrench(const std::string &body_name,
                                           const manifold::SE3::Wrench &wrench_body) {
        const int body_id = this->get_body_id(body_name);
        const manifold::SE3 T_wb = this->get_pose(body_id);

        // Rotate Body-Frame Force and Torque into World Frame.
        const Eigen::Vector3d f_world = T_wb.so3() * wrench_body.head<3>();
        const Eigen::Vector3d tau_world_at_origin = T_wb.so3() * wrench_body.tail<3>();

        // Lever Arm from Body Frame Origin (xpos) to CoM (xipos) in World Frame.
        const Eigen::Vector3d r_origin_to_com(
            this->data->xipos[(3 * body_id) + 0] - this->data->xpos[(3 * body_id) + 0],
            this->data->xipos[(3 * body_id) + 1] - this->data->xpos[(3 * body_id) + 1],
            this->data->xipos[(3 * body_id) + 2] - this->data->xpos[(3 * body_id) + 2]);

        // Translate Moment to CoM: tau_at_CoM = tau_at_origin - r_origin_to_CoM x f.
        const Eigen::Vector3d tau_world_at_com =
            tau_world_at_origin - r_origin_to_com.cross(f_world);

        // Write World-Frame Wrench into MuJoCo's xfrc_applied [fx fy fz tx ty tz].
        this->data->xfrc_applied[(6 * body_id) + 0] = f_world.x();
        this->data->xfrc_applied[(6 * body_id) + 1] = f_world.y();
        this->data->xfrc_applied[(6 * body_id) + 2] = f_world.z();
        this->data->xfrc_applied[(6 * body_id) + 3] = tau_world_at_com.x();
        this->data->xfrc_applied[(6 * body_id) + 4] = tau_world_at_com.y();
        this->data->xfrc_applied[(6 * body_id) + 5] = tau_world_at_com.z();
    }

    void Simulation::clear_external_wrenches() {
        mju_zero(this->data->xfrc_applied, 6 * this->model->nbody);
    }

    // --- Rendering & Visualisation ---

    void Simulation::render() {
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(this->window_, &viewport.width, &viewport.height);
        mjr_render(viewport, &this->scene_, &this->context_);
        glfwSwapBuffers(this->window_);
        glfwPollEvents();
    }

    void Simulation::update_scene() {
        mjv_updateScene(this->model, this->data, &this->option_, nullptr, &this->camera_, mjCAT_ALL,
                        &this->scene_);
    }

    void Simulation::update_model() const { mj_forward(this->model, this->data); }

    void Simulation::set_marker(const manifold::SE3 &pose) const {
        if (this->marker_id_ == -1) return;

        const Eigen::Vector3d pos = pose.r3();
        this->data->mocap_pos[(this->marker_id_ * 3) + 0] = pos.x();
        this->data->mocap_pos[(this->marker_id_ * 3) + 1] = pos.y();
        this->data->mocap_pos[(this->marker_id_ * 3) + 2] = pos.z();

        Eigen::Quaterniond quat = pose.so3().quat();
        this->data->mocap_quat[(this->marker_id_ * 4) + 0] = quat.w();
        this->data->mocap_quat[(this->marker_id_ * 4) + 1] = quat.x();
        this->data->mocap_quat[(this->marker_id_ * 4) + 2] = quat.y();
        this->data->mocap_quat[(this->marker_id_ * 4) + 3] = quat.z();
    }

    void Simulation::draw_arrow(const Eigen::Vector3d &point_origin,
                                const Eigen::Vector3d &point_end, const double width,
                                const float rgba[4]) {
        if (this->scene_.ngeom >= this->scene_.maxgeom) return;

        mjvGeom *geom = this->scene_.geoms + this->scene_.ngeom;
        this->scene_.ngeom++;

        mjv_initGeom(geom, mjGEOM_ARROW, nullptr, nullptr, nullptr, rgba);
        mjv_connector(geom, mjGEOM_ARROW, width, point_origin.data(), point_end.data());
    }

    // --- GLFW Mouse Callbacks ---

    void Simulation::mouse_button_callback(GLFWwindow *window, int button, int act, int mods) {
        if (!glfwGetWindowUserPointer(window)) return;
        g_button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
        g_button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
        g_button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
        glfwGetCursorPos(window, &g_last_x, &g_last_y);
    }

    void Simulation::mouse_move_callback(GLFWwindow *window, double x_pos, double y_pos) {
        auto *sim = static_cast<Simulation *>(glfwGetWindowUserPointer(window));
        if (!sim) return;

        double dx = x_pos - g_last_x;
        double dy = y_pos - g_last_y;
        g_last_x = x_pos;
        g_last_y = y_pos;

        if (!g_button_left && !g_button_middle && !g_button_right) return;

        int width;
        int height;
        glfwGetWindowSize(window, &width, &height);

        mjtMouse action = mjMOUSE_ZOOM;
        if (g_button_right) action = mjMOUSE_MOVE_V;
        else if (g_button_left) action = mjMOUSE_ROTATE_V;

        mjv_moveCamera(sim->model, action, dx / width, dy / height, &sim->scene_, &sim->camera_);
    }

    void Simulation::scroll_callback(GLFWwindow *window, double x_offset, double y_offset) {
        auto *sim = static_cast<Simulation *>(glfwGetWindowUserPointer(window));
        if (!sim) return;
        mjv_moveCamera(sim->model, mjMOUSE_ZOOM, 0, -sim->config_.scroll_sensitivity * y_offset,
                       &sim->scene_, &sim->camera_);
    }

}  // namespace xarm_geo
