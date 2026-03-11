#pragma once
// MuJoCo simulation wrapper. Loads any MJCF model, steps physics,
// reads body state, and optionally renders via GLFW.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include <mujoco/mujoco.h>

#include "types.h"
#include "gait.h"

namespace nmfly {

struct SimConfig {
    float    dt         = 0.0002f;  // physics timestep (seconds)
    bool     render     = false;    // open viewer window
    int      substeps   = 5;        // fast physics steps per controller step
    int      sync_every = 10;       // viewer syncs per N controller steps
    std::string model_path;         // MJCF file path (empty = use stick fly)
};

class Sim {
public:
    Sim() = default;
    ~Sim() { Shutdown(); }

    // Non-copyable, movable.
    Sim(const Sim&) = delete;
    Sim& operator=(const Sim&) = delete;

    bool Setup(const SimConfig& cfg, const std::string& xml_override = "") {
        cfg_ = cfg;

        // Load model.
        char error[1024] = {};
        if (!xml_override.empty()) {
            model_ = mj_loadXML(nullptr, nullptr, error, sizeof(error));
            // MuJoCo doesn't have a from_string in C API before 3.0.
            // Use mj_loadXML with a VFS.
            mjVFS vfs;
            mj_defaultVFS(&vfs);
            mj_addBufferVFS(&vfs, "model.xml",
                            xml_override.data(),
                            static_cast<int>(xml_override.size()));
            model_ = mj_loadXML("model.xml", &vfs, error, sizeof(error));
            mj_deleteVFS(&vfs);
        } else if (!cfg_.model_path.empty()) {
            // Detect format by extension.
            std::string ext;
            auto dot = cfg_.model_path.rfind('.');
            if (dot != std::string::npos)
                ext = cfg_.model_path.substr(dot);
            if (ext == ".mjb") {
                model_ = mj_loadModel(cfg_.model_path.c_str(), nullptr);
            } else {
                model_ = mj_loadXML(cfg_.model_path.c_str(), nullptr,
                                    error, sizeof(error));
            }
        } else {
            return false;  // caller should provide XML or path
        }

        if (!model_) {
            fprintf(stderr, "MuJoCo load error: %s\n", error);
            return false;
        }

        model_->opt.timestep = cfg_.dt;
        data_ = mj_makeData(model_);

        // Initialize visualization if requested.
#ifdef NMFLY_VIEWER
        if (cfg_.render) {
            if (!InitViewer()) {
                fprintf(stderr, "Warning: could not open viewer\n");
                cfg_.render = false;
            }
        }
#else
        cfg_.render = false;
#endif

        step_count_ = 0;
        ctrl_count_ = 0;
        return true;
    }

    // Step the simulation with the given joint targets.
    // targets: flat array of actuator control values, length = model_->nu.
    // Returns false if physics became unstable (auto-resets).
    bool Step(const float* targets, int n_targets) {
        if (!model_ || !data_) return false;

        // Apply control.
        int n = std::min(n_targets, model_->nu);
        std::memcpy(data_->ctrl, targets, n * sizeof(float));

        // One full step.
        mj_step(model_, data_);

        // Check for instability.
        bool stable = true;
        for (int i = 0; i < model_->nv && i < 10; ++i) {
            if (!std::isfinite(data_->qacc[i])) {
                stable = false;
                break;
            }
        }

        if (!stable) {
            mj_resetData(model_, data_);
            mj_forward(model_, data_);
            step_count_ = 0;
            ctrl_count_ = 0;
            return false;
        }

        // Sub-steps: advance physics with same control.
        for (int s = 1; s < cfg_.substeps; ++s) {
            mj_step(model_, data_);
        }

        step_count_ += cfg_.substeps;
        ctrl_count_++;

        // Sync viewer at reduced rate.
#ifdef NMFLY_VIEWER
        if (cfg_.render && viewer_running_ &&
            ctrl_count_ % cfg_.sync_every == 0) {
            SyncViewer();
        }
#endif

        return true;
    }

    // Read current body state.
    BodyState ReadState() const {
        BodyState s = {};
        if (!model_ || !data_) return s;

        int nq = std::min(42, model_->nq);
        for (int i = 0; i < nq; ++i)
            s.joint_angles[i] = static_cast<float>(data_->qpos[i]);

        int nv = std::min(42, model_->nv);
        for (int i = 0; i < nv; ++i)
            s.joint_velocities[i] = static_cast<float>(data_->qvel[i]);

        // Contact detection: check which leg geoms are in contact.
        for (int c = 0; c < data_->ncon; ++c) {
            int g1 = data_->contact[c].geom1;
            int g2 = data_->contact[c].geom2;
            for (int leg = 0; leg < 6; ++leg) {
                int gs = leg * 4;
                int ge = gs + 4;
                if ((gs <= g1 && g1 < ge) || (gs <= g2 && g2 < ge))
                    s.contacts[leg] = 1.0f;
            }
        }

        // Body velocity (first 6 DOF are the freejoint).
        if (model_->nv >= 6) {
            s.body_velocity[0] = static_cast<float>(data_->qvel[0]) * 1000.0f;
            s.body_velocity[1] = static_cast<float>(data_->qvel[1]) * 1000.0f;
            s.body_velocity[2] = static_cast<float>(data_->qvel[5]);
        }

        // Position (from freejoint qpos).
        if (model_->nq >= 3) {
            s.position[0] = static_cast<float>(data_->qpos[0]);
            s.position[1] = static_cast<float>(data_->qpos[1]);
            s.position[2] = static_cast<float>(data_->qpos[2]);
        }
        if (model_->nq >= 7) {
            // Extract yaw from quaternion (qpos[3..6]).
            double qw = data_->qpos[3], qz = data_->qpos[6];
            double qx = data_->qpos[4], qy = data_->qpos[5];
            s.heading = static_cast<float>(
                std::atan2(2.0 * (qw * qz + qx * qy),
                           1.0 - 2.0 * (qy * qy + qz * qz)));
        }

        s.step = step_count_;
        s.sim_time = step_count_ * cfg_.dt;
        return s;
    }

    bool IsRunning() const {
#ifdef NMFLY_VIEWER
        if (cfg_.render) return viewer_running_;
#endif
        return true;  // headless: always running
    }

    void Shutdown() {
#ifdef NMFLY_VIEWER
        ShutdownViewer();
#endif
        if (data_)  { mj_deleteData(data_);   data_ = nullptr; }
        if (model_) { mj_deleteModel(model_); model_ = nullptr; }
    }

    mjModel* model() { return model_; }
    mjData*  data()  { return data_; }
    int      step_count() const { return step_count_; }
    float    sim_time() const { return step_count_ * cfg_.dt; }
    const SimConfig& config() const { return cfg_; }
    void     add_steps(int n) { step_count_ += n; ctrl_count_++; }

    // Public viewer sync for direct-stepping code paths.
    void SyncViewerPublic() {
#ifdef NMFLY_VIEWER
        if (cfg_.render && viewer_running_) SyncViewer();
#endif
    }

private:
    SimConfig cfg_;
    mjModel*  model_ = nullptr;
    mjData*   data_  = nullptr;
    int       step_count_ = 0;
    int       ctrl_count_ = 0;

    // --- Viewer (GLFW) ---
#ifdef NMFLY_VIEWER
    bool viewer_running_ = false;

    // MuJoCo visualization objects.
    mjvCamera  cam_  = {};
    mjvOption  opt_  = {};
    mjvScene   scn_  = {};
    mjrContext con_  = {};

    struct GLFWwindow* window_ = nullptr;

    // Mouse interaction state.
    bool   mouse_left_  = false;
    bool   mouse_right_ = false;
    bool   mouse_mid_   = false;
    double mouse_x_ = 0, mouse_y_ = 0;

    bool InitViewer();
    void SyncViewer();
    void ShutdownViewer();

    static void MouseButtonCB(struct GLFWwindow* w, int button, int act, int mods);
    static void MouseMoveCB(struct GLFWwindow* w, double x, double y);
    static void ScrollCB(struct GLFWwindow* w, double dx, double dy);
#else
    void ShutdownViewer() {}
#endif
};

// --- GLFW viewer implementation ---
#ifdef NMFLY_VIEWER
#include <GLFW/glfw3.h>

// GLFW callbacks need to find the Sim instance via window user pointer.
inline void Sim::MouseButtonCB(GLFWwindow* w, int button, int act, int) {
    auto* s = static_cast<Sim*>(glfwGetWindowUserPointer(w));
    bool pressed = (act == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_LEFT)   s->mouse_left_  = pressed;
    if (button == GLFW_MOUSE_BUTTON_RIGHT)  s->mouse_right_ = pressed;
    if (button == GLFW_MOUSE_BUTTON_MIDDLE) s->mouse_mid_   = pressed;
}

inline void Sim::MouseMoveCB(GLFWwindow* w, double x, double y) {
    auto* s = static_cast<Sim*>(glfwGetWindowUserPointer(w));
    double dx = x - s->mouse_x_;
    double dy = y - s->mouse_y_;
    s->mouse_x_ = x;
    s->mouse_y_ = y;

    int width, height;
    glfwGetWindowSize(w, &width, &height);
    if (width == 0 || height == 0) return;

    if (s->mouse_left_) {
        // Rotate camera.
        mjv_moveCamera(s->model_, mjMOUSE_ROTATE_V, dx / width, dy / height,
                        &s->scn_, &s->cam_);
    } else if (s->mouse_right_) {
        // Translate camera.
        mjv_moveCamera(s->model_, mjMOUSE_MOVE_V, dx / width, dy / height,
                        &s->scn_, &s->cam_);
    } else if (s->mouse_mid_) {
        // Zoom.
        mjv_moveCamera(s->model_, mjMOUSE_ZOOM, 0, dy / height,
                        &s->scn_, &s->cam_);
    }
}

inline void Sim::ScrollCB(GLFWwindow* w, double, double dy) {
    auto* s = static_cast<Sim*>(glfwGetWindowUserPointer(w));
    mjv_moveCamera(s->model_, mjMOUSE_ZOOM, 0, -0.05 * dy,
                    &s->scn_, &s->cam_);
}

inline bool Sim::InitViewer() {
    if (!glfwInit()) return false;

    window_ = glfwCreateWindow(1280, 720, "nmfly-sim", nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(0);  // no vsync for max speed

    // Set up mouse callbacks.
    glfwSetWindowUserPointer(window_, this);
    glfwSetMouseButtonCallback(window_, MouseButtonCB);
    glfwSetCursorPosCallback(window_, MouseMoveCB);
    glfwSetScrollCallback(window_, ScrollCB);

    mjv_defaultCamera(&cam_);
    mjv_defaultOption(&opt_);
    mjv_defaultScene(&scn_);
    mjr_defaultContext(&con_);

    // Camera: follow the thorax from behind and above.
    // Find thorax body (NMF names it with "Thorax", stick fly uses "thorax").
    int thorax_id = 1;  // default: first body after world
    for (int i = 0; i < model_->nbody; ++i) {
        const char* name = mj_id2name(model_, mjOBJ_BODY, i);
        if (name) {
            std::string n(name);
            if (n.find("Thorax") != std::string::npos ||
                n.find("thorax") != std::string::npos) {
                thorax_id = i;
                break;
            }
        }
    }

    cam_.type = mjCAMERA_FREE;
    // Look at the model center from a reasonable distance.
    cam_.lookat[0] = model_->stat.center[0];
    cam_.lookat[1] = model_->stat.center[1];
    cam_.lookat[2] = model_->stat.center[2];
    cam_.distance = model_->stat.extent * 0.6;
    cam_.elevation = -20.0;
    cam_.azimuth = 135.0;  // front-left view

    mjv_makeScene(model_, &scn_, 5000);
    mjr_makeContext(model_, &con_, mjFONTSCALE_150);

    viewer_running_ = true;
    return true;
}

inline void Sim::SyncViewer() {
    if (!window_ || glfwWindowShouldClose(window_)) {
        viewer_running_ = false;
        return;
    }

    int width, height;
    glfwGetFramebufferSize(window_, &width, &height);
    mjrRect viewport = {0, 0, width, height};

    mjv_updateScene(model_, data_, &opt_, nullptr, &cam_,
                    mjCAT_ALL, &scn_);
    mjr_render(viewport, &scn_, &con_);

    glfwSwapBuffers(window_);
    glfwPollEvents();
}

inline void Sim::ShutdownViewer() {
    if (!window_) return;
    mjr_freeContext(&con_);
    mjv_freeScene(&scn_);
    glfwDestroyWindow(window_);
    glfwTerminate();
    window_ = nullptr;
    viewer_running_ = false;
}
#endif  // NMFLY_VIEWER

}  // namespace nmfly
