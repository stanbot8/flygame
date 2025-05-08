#pragma once
// MuJoCo simulation wrapper. Loads any MJCF model, steps physics,
// reads raw state, and optionally renders via GLFW.
//
// Body state extraction (ReadState) is NOT included here — each animal
// provides its own reader that knows its joint layout and contact geoms.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include <mujoco/mujoco.h>

#ifdef MJGAME_VIEWER
#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>
#endif

#include "types.h"

namespace mjgame {

struct SimConfig {
    float       dt         = 0.0005f;  // physics timestep (seconds)
    bool        render     = false;    // open viewer window
    int         substeps   = 5;        // fast physics steps per controller step
    int         sync_every = 5;        // viewer syncs per N controller steps
    std::string model_path;            // MJCF file path (empty = use xml_override)
    std::string window_title = "mujoco-game";
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
            mjVFS vfs;
            mj_defaultVFS(&vfs);
            mj_addBufferVFS(&vfs, "model.xml",
                            xml_override.data(),
                            static_cast<int>(xml_override.size()));
            model_ = mj_loadXML("model.xml", &vfs, error, sizeof(error));
            mj_deleteVFS(&vfs);
        } else if (!cfg_.model_path.empty()) {
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

#ifdef MJGAME_VIEWER
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

    // Step the simulation with the given actuator targets.
    // Returns false if physics became unstable (auto-resets).
    bool Step(const float* targets, int n_targets) {
        if (!model_ || !data_) return false;

        int n = std::min(n_targets, model_->nu);
        std::memcpy(data_->ctrl, targets, n * sizeof(float));

        mj_step(model_, data_);

        // Check for instability across all DOF.
        bool stable = true;
        for (int i = 0; i < model_->nv; ++i) {
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

#ifdef MJGAME_VIEWER
        if (cfg_.render && viewer_running_ &&
            ctrl_count_ % cfg_.sync_every == 0) {
            SyncViewer();
        }
#endif

        return true;
    }

    bool IsRunning() const {
#ifdef MJGAME_VIEWER
        if (cfg_.render) return viewer_running_;
#endif
        return true;  // headless: always running
    }

    void Shutdown() {
#ifdef MJGAME_VIEWER
        ShutdownViewer();
#endif
        if (data_)  { mj_deleteData(data_);   data_ = nullptr; }
        if (model_) { mj_deleteModel(model_); model_ = nullptr; }
    }

    // --- Accessors ---

    mjModel* model() { return model_; }
    mjData*  data()  { return data_; }
    const mjModel* model() const { return model_; }
    const mjData*  data()  const { return data_; }
    int64_t  step_count() const { return step_count_; }
    double   sim_time() const { return step_count_ * static_cast<double>(cfg_.dt); }
    const SimConfig& config() const { return cfg_; }
    void     add_steps(int64_t n) { step_count_ += n; ctrl_count_++; }

    // --- WASD key state (updated by GLFW key callback) ---

    bool KeyW() const { return key_w_; }
    bool KeyA() const { return key_a_; }
    bool KeyS() const { return key_s_; }
    bool KeyD() const { return key_d_; }
    bool KeySpace() const { return key_space_; }
    bool KeyShift() const { return key_shift_; }

    void PollKeys() {
#ifdef MJGAME_VIEWER
        if (!window_) return;
        key_w_     = glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS;
        key_a_     = glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS;
        key_s_     = glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS;
        key_d_     = glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS;
        key_space_ = glfwGetKey(window_, GLFW_KEY_SPACE) == GLFW_PRESS;
        key_shift_ = glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                     glfwGetKey(window_, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
#endif
    }

    // --- Viewer controls ---

    // HUD overlay text (set externally, rendered in top-left corner).
    std::string overlay_text;

    void SyncViewerPublic() {
#ifdef MJGAME_VIEWER
        if (cfg_.render && viewer_running_) SyncViewer();
#endif
    }

    struct GLFWwindow* glfw_window() const {
#ifdef MJGAME_VIEWER
        return window_;
#else
        return nullptr;
#endif
    }

    // Render MuJoCo scene to current framebuffer (no swap/poll).
    void RenderScene() {
#ifdef MJGAME_VIEWER
        if (!window_ || !model_ || !data_) return;
        int width, height;
        glfwGetFramebufferSize(window_, &width, &height);
        mjrRect viewport = {0, 0, width, height};
        mjv_updateScene(model_, data_, &opt_, nullptr, &cam_,
                        mjCAT_ALL, &scn_);
        mjr_render(viewport, &scn_, &con_);
#endif
    }

    // MuJoCo camera controls (for external mouse handling, e.g. ImGui).
    void RotateCamera(double dx, double dy, int w, int h) {
#ifdef MJGAME_VIEWER
        if (model_) mjv_moveCamera(model_, mjMOUSE_ROTATE_V, dx/w, dy/h, &scn_, &cam_);
#endif
    }
    void TranslateCamera(double dx, double dy, int w, int h) {
#ifdef MJGAME_VIEWER
        if (model_) mjv_moveCamera(model_, mjMOUSE_MOVE_V, dx/w, dy/h, &scn_, &cam_);
#endif
    }
    void ZoomCamera(double dy) {
#ifdef MJGAME_VIEWER
        if (model_) mjv_moveCamera(model_, mjMOUSE_ZOOM, 0, -0.05*dy, &scn_, &cam_);
#endif
    }

    // EMA smoothing factor for camera follow (0-1, lower = smoother).
    double camera_follow_alpha = 0.05;

    // Smooth camera follow (EMA on body position to remove jitter).
    void UpdateCameraFollow() {
#ifdef MJGAME_VIEWER
        if (!model_ || !data_) return;
        const double kCamAlpha = camera_follow_alpha;
        if (!lookat_init_) {
            for (int i = 0; i < 3; ++i)
                smooth_lookat_[i] = data_->qpos[i];
            lookat_init_ = true;
        } else {
            for (int i = 0; i < 3; ++i)
                smooth_lookat_[i] += kCamAlpha * (data_->qpos[i] - smooth_lookat_[i]);
        }
        cam_.lookat[0] = smooth_lookat_[0];
        cam_.lookat[1] = smooth_lookat_[1];
        cam_.lookat[2] = smooth_lookat_[2];
#endif
    }

private:
    SimConfig cfg_;
    mjModel*  model_ = nullptr;
    mjData*   data_  = nullptr;
    int64_t   step_count_ = 0;
    int64_t   ctrl_count_ = 0;

    // --- Viewer (GLFW) ---
#ifdef MJGAME_VIEWER
    bool viewer_running_ = false;

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

    // Keyboard state for WASD control.
    bool key_w_ = false, key_a_ = false, key_s_ = false, key_d_ = false;
    bool key_space_ = false, key_shift_ = false;

    // Smooth camera follow state.
    double smooth_lookat_[3] = {};
    bool lookat_init_ = false;

    bool InitViewer();
    void SyncViewer();
    void ShutdownViewer();

    static void MouseButtonCB(struct GLFWwindow* w, int button, int act, int mods);
    static void MouseMoveCB(struct GLFWwindow* w, double x, double y);
    static void ScrollCB(struct GLFWwindow* w, double dx, double dy);
    static void KeyCB(struct GLFWwindow* w, int key, int scancode, int act, int mods);
#else
    void ShutdownViewer() {}
    bool key_w_ = false, key_a_ = false, key_s_ = false, key_d_ = false;
    bool key_space_ = false, key_shift_ = false;
#endif
};

// --- GLFW viewer implementation ---
#ifdef MJGAME_VIEWER

inline void Sim::MouseButtonCB(GLFWwindow* w, int button, int act, int) {
    auto* s = static_cast<Sim*>(glfwGetWindowUserPointer(w));
    if (!s) return;
    bool pressed = (act == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_LEFT)   s->mouse_left_  = pressed;
    if (button == GLFW_MOUSE_BUTTON_RIGHT)  s->mouse_right_ = pressed;
    if (button == GLFW_MOUSE_BUTTON_MIDDLE) s->mouse_mid_   = pressed;
}

inline void Sim::MouseMoveCB(GLFWwindow* w, double x, double y) {
    auto* s = static_cast<Sim*>(glfwGetWindowUserPointer(w));
    if (!s) return;
    double dx = x - s->mouse_x_;
    double dy = y - s->mouse_y_;
    s->mouse_x_ = x;
    s->mouse_y_ = y;

    int width, height;
    glfwGetWindowSize(w, &width, &height);
    if (width == 0 || height == 0) return;

    if (s->mouse_left_) {
        mjv_moveCamera(s->model_, mjMOUSE_ROTATE_V, dx / width, dy / height,
                        &s->scn_, &s->cam_);
    } else if (s->mouse_right_) {
        mjv_moveCamera(s->model_, mjMOUSE_MOVE_V, dx / width, dy / height,
                        &s->scn_, &s->cam_);
    } else if (s->mouse_mid_) {
        mjv_moveCamera(s->model_, mjMOUSE_ZOOM, 0, dy / height,
                        &s->scn_, &s->cam_);
    }
}

inline void Sim::ScrollCB(GLFWwindow* w, double, double dy) {
    auto* s = static_cast<Sim*>(glfwGetWindowUserPointer(w));
    if (!s) return;
    mjv_moveCamera(s->model_, mjMOUSE_ZOOM, 0, -0.05 * dy,
                    &s->scn_, &s->cam_);
}

inline void Sim::KeyCB(GLFWwindow* w, int key, int, int act, int) {
    auto* s = static_cast<Sim*>(glfwGetWindowUserPointer(w));
    if (!s) return;
    bool pressed = (act != GLFW_RELEASE);
    switch (key) {
        case GLFW_KEY_W: s->key_w_ = pressed; break;
        case GLFW_KEY_A: s->key_a_ = pressed; break;
        case GLFW_KEY_S: s->key_s_ = pressed; break;
        case GLFW_KEY_D: s->key_d_ = pressed; break;
        case GLFW_KEY_SPACE:       s->key_space_ = pressed; break;
        case GLFW_KEY_LEFT_SHIFT:
        case GLFW_KEY_RIGHT_SHIFT: s->key_shift_ = pressed; break;
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(w, GLFW_TRUE);
            break;
        default: break;
    }
}

inline bool Sim::InitViewer() {
    if (!glfwInit()) return false;

    window_ = glfwCreateWindow(1280, 720, cfg_.window_title.c_str(),
                               nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(0);  // no vsync for max speed

    glfwSetWindowUserPointer(window_, this);
    glfwSetMouseButtonCallback(window_, MouseButtonCB);
    glfwSetCursorPosCallback(window_, MouseMoveCB);
    glfwSetScrollCallback(window_, ScrollCB);
    glfwSetKeyCallback(window_, KeyCB);

    mjv_defaultCamera(&cam_);
    mjv_defaultOption(&opt_);
    mjv_defaultScene(&scn_);
    mjr_defaultContext(&con_);

    // Camera: look at model center from a reasonable distance.
    cam_.type = mjCAMERA_FREE;
    cam_.lookat[0] = model_->stat.center[0];
    cam_.lookat[1] = model_->stat.center[1];
    cam_.lookat[2] = model_->stat.center[2];
    cam_.distance = model_->stat.extent * 1.2;
    cam_.elevation = -25.0;
    cam_.azimuth = 135.0;

    mjv_makeScene(model_, &scn_, 10000);
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

    // HUD overlay
    if (!overlay_text.empty()) {
        mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport,
                    overlay_text.c_str(), nullptr, &con_);
    }

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
#endif  // MJGAME_VIEWER

}  // namespace mjgame
