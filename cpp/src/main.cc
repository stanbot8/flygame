// flygame: Drosophila body simulation with embedded brain.
//
// MuJoCo fly body driven by a spiking neural network brain.
// WASD keyboard control feeds intent through descending neurons.
// UI matches the fwmc brain viewer (ImGui panels + brain point cloud).
//
// Usage:
//   nmfly-sim                          # standalone, walk forward
//   nmfly-sim --brain 127.0.0.1:9100   # connect to FWMC brain
//   nmfly-sim --brain-local FILE       # embedded brain from .brain spec
//   nmfly-sim --headless               # no viewer, max speed

// Include glad BEFORE anything that pulls in GL headers (MuJoCo, GLFW).
#ifdef NMFLY_VIEWER
#include <glad/gl.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <chrono>
#include <filesystem>
#include <string>
#include <charconv>

#include <atomic>
#include <mutex>
#include <thread>

#include "types.h"
#include "sim.h"
#include "tcp.h"
#include "nmf_cpg.h"
#include "proprio.h"
#include "fly_state.h"
#include "fly_tcp.h"
#include "brain.h"

#ifdef NMFLY_VIEWER
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include "brain_viewport.h"
#endif

using namespace mjgame;
using namespace nmfly;
using Clock = std::chrono::steady_clock;

constexpr int kMaxRegions = 16;

struct AppConfig {
    std::string brain_host;
    int         brain_port  = 9100;
    bool        standalone  = true;
    bool        headless    = false;
    float       dt          = 0.0002f;
    int         substeps    = 5;
    float       duration    = 0.0f;    // 0 = run forever
    std::string model_path;
    float       fwd_speed   = 15.0f;  // standalone forward speed mm/s

    // Embedded brain mode
    bool        use_brain   = false;
    std::string brain_spec;           // path to .brain file

    // Automated test mode
    bool        test_mode   = false;
};

static void PrintUsage(const char* prog) {
    printf("flygame: Drosophila body simulation with embedded brain\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --brain HOST:PORT   Connect to brain sim (default: standalone)\n");
    printf("  --model PATH        MJCF model file (default: flygym NMF)\n");
    printf("  --headless          No viewer window (max speed)\n");
    printf("  --dt SECONDS        Physics timestep (default: 0.0002)\n");
    printf("  --substeps N        Physics sub-steps per controller step (default: 5)\n");
    printf("  --duration SECONDS  Run for N seconds sim time (default: forever)\n");
    printf("  --speed MM_S        Standalone forward speed (default: 15)\n");
#ifdef NMFLY_BRAIN
    printf("  --brain-local FILE  Embedded brain from .brain spec (WASD control)\n");
#endif
    printf("  --help              Show this help\n");
}

static bool ParseArgs(int argc, char** argv, AppConfig& cfg) {
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            PrintUsage(argv[0]);
            return false;
        }
        if (strcmp(arg, "--brain") == 0 && i + 1 < argc) {
            std::string val = argv[++i];
            auto colon = val.rfind(':');
            if (colon != std::string::npos) {
                cfg.brain_host = val.substr(0, colon);
                std::from_chars(val.data() + colon + 1,
                                val.data() + val.size(),
                                cfg.brain_port);
            } else {
                cfg.brain_host = val;
            }
            cfg.standalone = false;
        }
        else if (strcmp(arg, "--model") == 0 && i + 1 < argc) {
            cfg.model_path = argv[++i];
        }
        else if (strcmp(arg, "--headless") == 0) {
            cfg.headless = true;
        }
        else if (strcmp(arg, "--dt") == 0 && i + 1 < argc) {
            cfg.dt = std::strtof(argv[++i], nullptr);
        }
        else if (strcmp(arg, "--substeps") == 0 && i + 1 < argc) {
            std::from_chars(argv[i + 1],
                            argv[i + 1] + strlen(argv[i + 1]),
                            cfg.substeps);
            ++i;
        }
        else if (strcmp(arg, "--duration") == 0 && i + 1 < argc) {
            cfg.duration = std::strtof(argv[++i], nullptr);
        }
        else if (strcmp(arg, "--speed") == 0 && i + 1 < argc) {
            cfg.fwd_speed = std::strtof(argv[++i], nullptr);
        }
#ifdef NMFLY_BRAIN
        else if (strcmp(arg, "--brain-local") == 0 && i + 1 < argc) {
            cfg.brain_spec = argv[++i];
            cfg.use_brain = true;
            cfg.standalone = false;
        }
#endif
        else if (strcmp(arg, "--test") == 0) {
            cfg.test_mode = true;
            cfg.headless = true;
            cfg.use_brain = true;
            cfg.standalone = false;
            cfg.duration = 3.0f;
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", arg);
            PrintUsage(argv[0]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    AppConfig cfg;
    if (!ParseArgs(argc, argv, cfg)) return 1;

    const char* mode_str = cfg.standalone ? "standalone" :
                           cfg.use_brain ? "embedded-brain" : "tcp-brain";
    printf("[flygame] Starting (dt=%.5f, substeps=%d, %s)\n",
           cfg.dt, cfg.substeps, mode_str);

    // Set up simulation.
    SimConfig sim_cfg;
    sim_cfg.dt         = cfg.dt;
    sim_cfg.render     = !cfg.headless;
    sim_cfg.substeps   = cfg.substeps;
    sim_cfg.sync_every = 999999;  // disable auto-sync (ImGui handles rendering)
    sim_cfg.model_path = cfg.model_path;

    // Auto-detect MuJoCo plugin directory (for STL/OBJ mesh decoders).
    {
        namespace fs = std::filesystem;
        fs::path exe_dir = fs::path(argv[0]).parent_path();
        // Check next to exe first, then pip install location.
        fs::path candidates[] = {
            exe_dir / "plugin",
            exe_dir / ".." / "plugin",
        };
        for (auto& p : candidates) {
            if (fs::exists(p / "stl_decoder.dll") || fs::exists(p / "stl_decoder.so")) {
                sim_cfg.plugin_dir = fs::canonical(p).string();
                break;
            }
        }
    }

    Sim sim;
    if (cfg.model_path.empty()) {
        // Look for the pre-exported NMF model (with actuators) next to the exe.
        namespace fs = std::filesystem;
        fs::path exe_dir = fs::path(argv[0]).parent_path();
        fs::path search[] = {
            exe_dir / "data" / "nmf_complete.xml",
            exe_dir / ".." / "data" / "nmf_complete.xml",
            exe_dir / ".." / ".." / "cpp" / "data" / "nmf_complete.xml",
        };
        for (auto& p : search) {
            if (fs::exists(p)) {
                cfg.model_path = fs::canonical(p).string();
                sim_cfg.model_path = cfg.model_path;
                break;
            }
        }
        if (cfg.model_path.empty()) {
            fprintf(stderr, "[flygame] NMF model not found. Place nmf_complete.xml in data/\n");
            return 1;
        }
        printf("[flygame] Using NMF model: %s\n", cfg.model_path.c_str());
    }

    if (!sim.Setup(sim_cfg)) {
        fprintf(stderr, "[flygame] Failed to set up simulation\n");
        return 1;
    }
    printf("[flygame] MuJoCo ready (%d actuators, %d DOF)\n",
           static_cast<int>(sim.model()->nu),
           static_cast<int>(sim.model()->nv));

    // Load glad on the body window (needed for ImGui + brain FBO).
#ifdef NMFLY_VIEWER
    GLFWwindow* body_window = sim.glfw_window();
    GLFWwindow* brain_window = nullptr;
    BrainViewport brain_vp;

    if (body_window && !cfg.headless) {
        glfwMakeContextCurrent(body_window);
        gladLoadGL(glfwGetProcAddress);

        // Create hidden brain GL context (GL 3.3 core, shared with body).
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        brain_window = glfwCreateWindow(2, 2, "Brain", nullptr, body_window);
        if (brain_window) {
            glfwMakeContextCurrent(brain_window);
            gladLoadGL(glfwGetProcAddress);
            brain_vp.Init();
            glfwMakeContextCurrent(body_window);
        }

        // Initialize ImGui on body window.
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForOpenGL(body_window, true);
        ImGui_ImplOpenGL3_Init("#version 130");
        ImGui::StyleColorsDark();
        printf("[flygame] ImGui + brain viewport ready\n");
    }
#endif

    // NMF initialization: set qpos to tripod pose, then settle with
    // actuators holding that pose (no qpos tracking drift).
    int nu = sim.model()->nu;
    NmfCpg nmf_cpg;
    nmf_cpg.Init(12.0, 0.0);

    // Get the standing neutral from CPG (tripod base + zero amplitude).
    double walk_ctrl[48];
    nmf_cpg.GetCtrl(walk_ctrl);

    // Set qpos directly to the tripod pose.
    for (int i = 0; i < 42; ++i) {
        int jnt_id = sim.model()->actuator_trnid[2 * i];
        if (jnt_id >= 0 && jnt_id < sim.model()->njnt) {
            int qadr = sim.model()->jnt_qposadr[jnt_id];
            sim.data()->qpos[qadr] = walk_ctrl[i];
        }
    }
    for (int i = 0; i < sim.model()->nv; ++i)
        sim.data()->qvel[i] = 0.0;

    // Set ctrl to match: actuators hold the tripod pose.
    for (int i = 0; i < 42; ++i)
        sim.data()->ctrl[i] = walk_ctrl[i];
    for (int i = 42; i < 48; ++i)
        sim.data()->ctrl[i] = 1.0;  // adhesion on (flygym uses 0/1)

    mj_forward(sim.model(), sim.data());
    printf("[flygame] NMF init (tripod pose): z=%.4f, ncon=%d\n",
           sim.data()->qpos[2], sim.data()->ncon);

    // Warmup: hold tripod pose via actuators while contacts settle.
    for (int s = 0; s < 500; ++s) {
        mj_step(sim.model(), sim.data());
        if (s < 3 || s == 499)
            printf("[flygame]   warmup %d: z=%.4f, ncon=%d\n",
                   s, sim.data()->qpos[2], sim.data()->ncon);
#ifdef NMFLY_VIEWER
        if (s % 100 == 0 && body_window)
            sim.SyncViewerPublic();
#endif
    }
    sim.SyncViewerPublic();
    printf("[flygame] NMF settled (z=%.4f, ncon=%d)\n",
           sim.data()->qpos[2], sim.data()->ncon);

    // Start walking in standalone mode.
    if (cfg.standalone)
        nmf_cpg.SetDrive(1.0, 0.0);

    // Initialize embedded brain. Auto-enable when a .brain spec is found.
#ifdef NMFLY_BRAIN
    Brain brain;
    if (!cfg.use_brain && cfg.standalone) {
        // Auto-detect brain spec file — enable brain automatically.
        namespace fs = std::filesystem;
        fs::path exe_dir = fs::path(argv[0]).parent_path();
        fs::path search[] = {
            exe_dir / "data" / "drosophila_full.brain",
            exe_dir / ".." / "data" / "drosophila_full.brain",
        };
        for (auto& p : search) {
            if (fs::exists(p)) {
                cfg.brain_spec = fs::canonical(p).string();
                cfg.use_brain = true;
                cfg.standalone = false;
                printf("[flygame] Auto-detected brain: %s\n", cfg.brain_spec.c_str());
                break;
            }
        }
    }
    if (cfg.use_brain) {
        // Find brain spec file
        std::string spec = cfg.brain_spec;
        if (spec.empty()) {
            namespace fs = std::filesystem;
            fs::path exe_dir = fs::path(argv[0]).parent_path();
            fs::path search[] = {
                exe_dir / "data" / "drosophila_full.brain",
                exe_dir / ".." / "data" / "drosophila_full.brain",
            };
            for (auto& p : search) {
                if (fs::exists(p)) {
                    spec = fs::canonical(p).string();
                    break;
                }
            }
        }
        if (spec.empty()) {
            fprintf(stderr, "[flygame] No .brain spec found for --brain-local\n");
            return 1;
        }
        printf("[flygame] Loading brain spec: %s\n", spec.c_str());
        if (!brain.Init(spec)) {
            fprintf(stderr, "[flygame] Brain init failed\n");
            return 1;
        }
        printf("[flygame] Brain active: WASD to control\n");

        // Build brain point cloud from neuron positions.
#ifdef NMFLY_VIEWER
        if (brain_window && brain.IsInitialized()) {
            glfwMakeContextCurrent(brain_window);
            brain_vp.BuildFromSDF();
            glfwMakeContextCurrent(body_window);
            printf("[flygame] Brain viewport: %zu points\n",
                   brain_vp.PointCount());
        }
#endif
    }
#endif

    // Connect to brain if requested.
    TcpClient tcp;
    if (!cfg.standalone && !cfg.use_brain) {
        printf("[flygame] Connecting to brain at %s:%d...\n",
               cfg.brain_host.c_str(), cfg.brain_port);
        if (tcp.Connect(cfg.brain_host, cfg.brain_port)) {
            printf("[flygame] Connected to brain\n");
        } else {
            printf("[flygame] Could not connect. Running standalone.\n");
            cfg.standalone = true;
        }
    }

    // Brain thread: decouples brain sim from physics loop so physics
    // runs at full speed while the brain catches up asynchronously.
    std::atomic<bool> brain_running{false};
    std::thread brain_thread;
    std::mutex brain_mutex;
    WasdIntent brain_intent;        // written by main, read by brain thread
    MotorCommand brain_cmd;         // written by brain thread, read by main
    SensoryReading brain_proprio[kTotalChannels] = {};  // body -> brain
    int brain_proprio_count = 0;
    int brain_spike_count = 0;
    std::array<float, kMaxRegions> brain_frame_counts = {};
    bool brain_has_new_data = false;

#ifdef NMFLY_BRAIN
    if (cfg.use_brain && brain.IsInitialized()) {
        brain_running = true;
        brain_thread = std::thread([&]() {
            while (brain_running) {
                WasdIntent intent;
                SensoryReading proprio[kTotalChannels];
                int proprio_n = 0;
                {
                    std::lock_guard<std::mutex> lk(brain_mutex);
                    intent = brain_intent;
                    proprio_n = brain_proprio_count;
                    if (proprio_n > 0)
                        std::memcpy(proprio, brain_proprio, proprio_n * sizeof(SensoryReading));
                }
                brain.SetProprio(proprio, proprio_n);
                brain.SetIntent(intent);
                brain.Step(1.0f);

                // Publish results.
                auto bcmd = brain.GetMotorCommand();
                int spikes = brain.CountSpikes();
                auto& neurons = brain.Neurons();
                std::array<float, kMaxRegions> fc = {};
                for (size_t i = 0; i < neurons.n; ++i) {
                    int r = neurons.region[i];
                    if (r >= 0 && r < kMaxRegions) fc[r] += neurons.spiked[i];
                }
                {
                    std::lock_guard<std::mutex> lk(brain_mutex);
                    brain_cmd = bcmd;
                    brain_spike_count = spikes;
                    brain_frame_counts = fc;
                    brain_has_new_data = true;
                }
            }
        });
        printf("[flygame] Brain thread started\n");
    }
#endif

    MotorCommand cmd;
    if (cfg.standalone) {
        cmd.forward_velocity = cfg.fwd_speed;
    }

    // Sensory buffer.
    SensoryReading sensory[kTotalChannels];
    ProprioConfig proprio_cfg;

    // Smoothed motor command (EMA filter to prevent sudden jumps).
    MotorCommand smooth_cmd;
    float cmd_tau = 0.05f;  // 50ms smoothing time constant

    // Exchange rate: how often to send/receive over TCP.
    int exchange_every = 20;
    int ctrl_count = 0;

    // Performance tracking.
    auto wall_start = Clock::now();
    int  total_steps = 0;

    // Per-region spike rates for display.
    constexpr int kRasterFrames = 200;
    std::vector<std::array<float, kMaxRegions>> raster_history(kRasterFrames);
    int raster_head = 0;
    float region_rates[kMaxRegions] = {};
    int region_neuron_count[kMaxRegions] = {};
    int spike_count_display = 0;
    int n_display_regions = 0;

#ifdef NMFLY_BRAIN
    if (brain.IsInitialized()) {
        n_display_regions = std::min(brain.RegionCount(), kMaxRegions);
        auto& neurons = brain.Neurons();
        for (size_t i = 0; i < neurons.n; ++i) {
            int r = neurons.region[i];
            if (r >= 0 && r < kMaxRegions) region_neuron_count[r]++;
        }
    }
#endif

    // FPS tracking.
    double fps_time = 0.0;
    int fps_count = 0;
    float fps_display = 0.0f;
#ifdef NMFLY_VIEWER
    if (body_window) fps_time = glfwGetTime();
#endif

    // Mouse state for body camera.
    bool body_mouse_left = false, body_mouse_right = false, body_mouse_mid = false;
    double body_last_mx = 0, body_last_my = 0;

    // Region visibility/size for brain overlay.
    float region_size[14] = {1,2,2,4,4,4,4,4,4,4,4,4,4,4};
    bool region_dirty = false;
    bool regions_open = false;

    // Test mode: track positions at phase boundaries.
    // Phases: 0=idle(0-0.5), 1=W(0.5-1.5), 2=W+A(1.5-2.0), 3=W+D(2.0-2.5), 4=idle(2.5-3.0)
    float test_pos[6][3] = {};  // x,y,heading at each phase boundary (idx 0 = start)
    int   test_phase = 0;
    float test_phase_times[] = {0.0f, 0.5f, 1.5f, 2.0f, 2.5f, 3.0f};

    auto get_heading = [&]() -> float {
        // Extract yaw from free-joint quaternion (qpos[3..6] = w,x,y,z)
        double qw = sim.data()->qpos[3], qx = sim.data()->qpos[4];
        double qy = sim.data()->qpos[5], qz = sim.data()->qpos[6];
        return static_cast<float>(std::atan2(2.0*(qw*qz + qx*qy),
                                              1.0 - 2.0*(qy*qy + qz*qz)));
    };

    printf("[flygame] Running%s...\n",
           cfg.headless ? " (headless)" : " (with viewer)");

    // Main loop.
    while (sim.IsRunning()) {
        // Duration limit.
        if (cfg.duration > 0.0f && sim.sim_time() >= cfg.duration)
            break;

#ifdef NMFLY_VIEWER
        if (body_window) {
            glfwPollEvents();
            sim.PollKeys();
            if (glfwWindowShouldClose(body_window))
                break;

            // FPS
            fps_count++;
            double now = glfwGetTime();
            if (now - fps_time >= 0.5) {
                fps_display = static_cast<float>(fps_count / (now - fps_time));
                fps_count = 0;
                fps_time = now;
            }
        }
#endif

        // Embedded brain: WASD -> brain -> motor command.
#ifdef NMFLY_BRAIN
        if (cfg.use_brain && brain.IsInitialized()) {
            // Read WASD from viewer keyboard state
            constexpr float kRiseRate = 0.45f;
            constexpr float kDecayRate = 0.25f;
            constexpr float kBrakeRate = 0.30f;
            static float smooth_fwd = 0.0f;
            static float smooth_turn = 0.0f;

            // In test mode, inject synthetic WASD input sequence:
            //   0-0.5s: idle (should stay still)
            //   0.5-1.5s: W (forward, should walk straight)
            //   1.5-2.0s: W+A (forward + left turn)
            //   2.0-2.5s: W+D (forward + right turn)
            //   2.5-3.0s: idle (should stop)
            bool test_w = false, test_a = false, test_d = false;
            if (cfg.test_mode) {
                float t = sim.sim_time();
                if (t >= 0.5 && t < 2.0) test_w = true;
                if (t >= 1.5 && t < 2.0) test_a = true;
                if (t >= 2.0 && t < 2.5) { test_w = true; test_d = true; }
            }

            float target_fwd = (sim.KeyW() || test_w) ? 1.0f : 0.0f;
            if (sim.KeyS()) { smooth_fwd -= kBrakeRate; target_fwd = 0.0f; }
            float target_turn = 0.0f;
            if (sim.KeyA() || test_a) target_turn -= 1.0f;
            if (sim.KeyD() || test_d) target_turn += 1.0f;

            float fwd_rate = (target_fwd > smooth_fwd) ? kRiseRate : kDecayRate;
            smooth_fwd += (target_fwd - smooth_fwd) * fwd_rate;
            smooth_fwd = std::clamp(smooth_fwd, 0.0f, 1.0f);
            float turn_rate = (std::abs(target_turn) > std::abs(smooth_turn))
                              ? kRiseRate : kDecayRate;
            smooth_turn += (target_turn - smooth_turn) * turn_rate;
            smooth_turn = std::clamp(smooth_turn, -1.0f, 1.0f);
            if (std::abs(smooth_fwd) < 0.01f) smooth_fwd = 0.0f;
            if (std::abs(smooth_turn) < 0.01f) smooth_turn = 0.0f;

            // Send intent to brain thread.
            {
                std::lock_guard<std::mutex> lk(brain_mutex);
                brain_intent.forward = smooth_fwd;
                brain_intent.turn = smooth_turn;
                brain_intent.running = sim.KeySpace();
                brain_intent.flying = sim.KeyShift();
            }

            // Read latest motor command from brain thread.
            bool has_input = (smooth_fwd > 0.01f) ||
                             (std::abs(smooth_turn) > 0.01f);
            {
                std::lock_guard<std::mutex> lk(brain_mutex);
                cmd = brain_cmd;
                if (brain_has_new_data) {
                    spike_count_display = brain_spike_count;
                    std::array<float, kMaxRegions> frame_counts = brain_frame_counts;
                    for (int r = 0; r < kMaxRegions; ++r) {
                        if (region_neuron_count[r] > 0)
                            frame_counts[r] /= static_cast<float>(region_neuron_count[r]);
                        region_rates[r] = frame_counts[r] * 1000.0f;
                    }
                    raster_history[raster_head] = frame_counts;
                    raster_head = (raster_head + 1) % kRasterFrames;
                    brain_has_new_data = false;
                }
            }
            // Gate motor output on WASD input.
            if (!has_input && !brain_intent.flying) {
                cmd.forward_velocity = 0.0f;
            }
            // Angular velocity: use direct WASD for responsive, reliable
            // turning. Brain's VNC L/R asymmetry is too noisy/indirect
            // for clean turning control.
            cmd.angular_velocity = smooth_turn * 3.0f;

            // Update brain point cloud visualization.
#ifdef NMFLY_VIEWER
            if (brain_window && ctrl_count % 2 == 0) {
                glfwMakeContextCurrent(brain_window);
                brain_vp.UpdateActivityByRegion(region_rates);
                glfwMakeContextCurrent(body_window);
            }
#endif
        }
#endif

        // TCP exchange at reduced rate.
        if (!cfg.use_brain && !cfg.standalone && tcp.IsConnected() &&
            ctrl_count % exchange_every == 0) {
            BodyState state = ReadFlyState(sim);
            int n_sensory = EncodeProprioception(state, sensory,
                                                 kTotalChannels, proprio_cfg);
            MotorCommand brain_cmd;
            if (tcp.Exchange(sensory, n_sensory, brain_cmd)) {
                brain_cmd.freeze = 0.0f;  // body sim handles freeze
                cmd = brain_cmd;
            }
        }

        // Smooth motor command to prevent sudden jumps.
        {
            float step_dt = cfg.dt * cfg.substeps;
            float a = 1.0f - std::exp(-step_dt / cmd_tau);
            smooth_cmd.forward_velocity += a * (cmd.forward_velocity - smooth_cmd.forward_velocity);
            smooth_cmd.angular_velocity += a * (cmd.angular_velocity - smooth_cmd.angular_velocity);
        }

        // Map gait targets to MuJoCo actuators.
        {
            // NeuroMechFly: Kuramoto CPG with real trajectory lookup.
            // Clamp magnitude to 1.0: above that, leg amplitudes get too
            // large and launch the fly off the ground. The NMF model's
            // joint trajectories are calibrated for magnitude <= 1.0.
            double magnitude = std::clamp(
                static_cast<double>(smooth_cmd.forward_velocity) / 15.0, 0.0, 1.0);
            double turn = std::clamp(
                static_cast<double>(smooth_cmd.angular_velocity) / 3.0, -1.0, 1.0);
            nmf_cpg.SetDrive(magnitude, turn);

            bool stable = true;
            double sub_dt = static_cast<double>(cfg.dt);
            for (int s = 0; s < cfg.substeps; ++s) {
                nmf_cpg.Step(sub_dt);
                double nmf_ctrl[48];
                nmf_cpg.GetCtrl(nmf_ctrl);
                for (int i = 0; i < 48; ++i)
                    sim.data()->ctrl[i] = nmf_ctrl[i];
                mj_step(sim.model(), sim.data());
                for (int j = 0; j < sim.model()->nv; ++j) {
                    if (!std::isfinite(sim.data()->qacc[j])) {
                        stable = false;
                        break;
                    }
                }
                if (!stable) break;
            }
            if (!stable) {
                // Reset to tripod pose (not default stretch).
                mj_resetData(sim.model(), sim.data());
                nmf_cpg.Init(12.0, 0.0);
                double reset_ctrl[48];
                nmf_cpg.GetCtrl(reset_ctrl);
                for (int i = 0; i < 42; ++i) {
                    int jnt_id = sim.model()->actuator_trnid[2 * i];
                    if (jnt_id >= 0 && jnt_id < sim.model()->njnt) {
                        int qadr = sim.model()->jnt_qposadr[jnt_id];
                        sim.data()->qpos[qadr] = reset_ctrl[i];
                    }
                    sim.data()->ctrl[i] = reset_ctrl[i];
                }
                for (int i = 42; i < 48; ++i)
                    sim.data()->ctrl[i] = 20.0;
                mj_forward(sim.model(), sim.data());
                smooth_cmd = {};
                cmd = cfg.standalone ? MotorCommand{cfg.fwd_speed} : MotorCommand();
                if (cfg.standalone) nmf_cpg.SetDrive(1.0, 0.0);
                printf("[flygame] Physics reset (instability)\n");
                ctrl_count = 0;
                continue;
            }

            sim.add_steps(cfg.substeps);
            total_steps += cfg.substeps;
            ctrl_count++;

            // Update proprioceptive feedback for brain thread.
#ifdef NMFLY_BRAIN
            if (cfg.use_brain && ctrl_count % 5 == 0) {
                BodyState body_state = ReadFlyState(sim);
                SensoryReading proprio_buf[kTotalChannels];
                int n_proprio = EncodeProprioception(body_state, proprio_buf,
                                                      kTotalChannels, proprio_cfg);
                std::lock_guard<std::mutex> lk(brain_mutex);
                std::memcpy(brain_proprio, proprio_buf, n_proprio * sizeof(SensoryReading));
                brain_proprio_count = n_proprio;
            }
#endif

            if (ctrl_count <= 5 || ctrl_count % 500 == 0)
                printf("[flygame] frame %d: z=%.4f ncon=%d\n",
                       ctrl_count, sim.data()->qpos[2], sim.data()->ncon);

            // Test mode: sample position at phase boundaries.
            if (cfg.test_mode && test_phase < 6) {
                float t = sim.sim_time();
                if (t >= test_phase_times[test_phase]) {
                    test_pos[test_phase][0] = static_cast<float>(sim.data()->qpos[0]);
                    test_pos[test_phase][1] = static_cast<float>(sim.data()->qpos[1]);
                    test_pos[test_phase][2] = get_heading();
                    printf("[test] phase %d (t=%.2f): x=%.4f y=%.4f heading=%.4f\n",
                           test_phase, t,
                           test_pos[test_phase][0],
                           test_pos[test_phase][1],
                           test_pos[test_phase][2]);
                    test_phase++;
                }
            }

            // Render frame.
#ifdef NMFLY_VIEWER
            if (body_window && !cfg.headless) {
                glfwMakeContextCurrent(body_window);

                // Follow camera.
                sim.UpdateCameraFollow();

                // Render MuJoCo scene.
                int fb_w, fb_h;
                glfwGetFramebufferSize(body_window, &fb_w, &fb_h);
                glViewport(0, 0, fb_w, fb_h);
                glClearColor(0.2f, 0.2f, 0.3f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                sim.RenderScene();

                // Render brain FBO.
                if (brain_window && brain_vp.initialized) {
                    glfwMakeContextCurrent(brain_window);
                    brain_vp.Render(512, 384);
                    glfwMakeContextCurrent(body_window);
                }

                // ImGui frame.
                ImGui_ImplOpenGL3_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();

                int win_w, win_h;
                glfwGetWindowSize(body_window, &win_w, &win_h);
                const bool show_panel = (win_w > 500 && win_h > 400);

                if (show_panel) {
                    // Side panel (matches fwmc viewer layout).
                    char title_buf[64];
                    snprintf(title_buf, sizeof(title_buf),
                             "flygame | %.0f FPS###main", fps_display);
                    float panel_h = static_cast<float>(win_h) - 20.0f;
                    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
                    ImGui::SetNextWindowSize(ImVec2(300, panel_h), ImGuiCond_Always);
                    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6, 4));
                    ImGui::Begin(title_buf);

                    // -- Simulation header --
                    if (ImGui::CollapsingHeader("Simulation",
                            ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::PushID("sim");

#ifdef NMFLY_BRAIN
                        if (brain.IsInitialized()) {
                            ImGui::Text("Neurons: %zuK | Synapses: %.1fM",
                                        brain.NeuronCount() / 1000,
                                        brain.SynapseCount() / 1e6f);
                            ImGui::Text("Spikes: %d (%.2f%%)",
                                        spike_count_display,
                                        brain.SpikeRate() * 100.0f);
                            ImGui::Text("Motor L/R: %.1f / %.1f Hz",
                                        brain.MotorRateLeft(),
                                        brain.MotorRateRight());
                            ImGui::Text("Brain time: %.0f ms",
                                        brain.SimTimeMs());

                            auto bcmd = brain.GetMotorCommand();
                            ImGui::Text("Cmd: fwd=%.1f mm/s  turn=%.2f rad/s",
                                        bcmd.forward_velocity,
                                        bcmd.angular_velocity);
                        }
#endif
                        ImGui::Text("Physics: z=%.4f  ncon=%d",
                                    sim.data()->qpos[2], sim.data()->ncon);
                        ImGui::Text("Mode: %s", mode_str);

                        ImGui::PopID();
                    }

                    // -- Activity Monitor --
#ifdef NMFLY_BRAIN
                    if (brain.IsInitialized() &&
                        ImGui::CollapsingHeader("Activity Monitor",
                            ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::PushID("activity");

                        // Per-region firing rates (dynamic from brain spec).
                        if (ImGui::TreeNodeEx("Firing Rates",
                                ImGuiTreeNodeFlags_DefaultOpen |
                                ImGuiTreeNodeFlags_SpanAvailWidth)) {
                            for (int r = 0; r < n_display_regions; ++r) {
                                if (region_neuron_count[r] == 0) continue;
                                float rate = region_rates[r];
                                ImVec4 col = (rate < 1.0f)
                                    ? ImVec4(0.5f, 0.5f, 0.5f, 1.0f)
                                    : (rate < 30.0f)
                                    ? ImVec4(0.3f, 1.0f, 0.3f, 1.0f)
                                    : (rate < 60.0f)
                                    ? ImVec4(1.0f, 1.0f, 0.3f, 1.0f)
                                    : ImVec4(1.0f, 0.3f, 0.3f, 1.0f);
                                ImGui::TextColored(col, "%-5s %6.1f Hz",
                                                   brain.RegionName(r), rate);
                            }
                            ImGui::TreePop();
                        }

                        // Spike raster.
                        if (ImGui::TreeNodeEx("Spike Raster",
                                ImGuiTreeNodeFlags_DefaultOpen |
                                ImGuiTreeNodeFlags_SpanAvailWidth)) {
                            ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
                            ImVec2 canvas_size = ImVec2(
                                ImGui::GetContentRegionAvail().x, 80.0f);
                            ImGui::InvisibleButton("##raster", canvas_size);
                            ImDrawList* draw = ImGui::GetWindowDrawList();

                            float col_w = canvas_size.x / kRasterFrames;
                            for (int f = 0; f < kRasterFrames; ++f) {
                                int idx = (raster_head + f) % kRasterFrames;
                                float total_val = 0.0f;
                                int nr = std::max(n_display_regions, 1);
                                for (int r = 0; r < nr; ++r)
                                    total_val += raster_history[idx][r];
                                total_val = std::clamp(
                                    total_val / static_cast<float>(nr), 0.0f, 1.0f);
                                float brightness = std::clamp(
                                    total_val * 3.0f, 0.0f, 1.0f);
                                ImU32 color = ImGui::GetColorU32(ImVec4(
                                    brightness,
                                    brightness * 0.7f,
                                    brightness * 0.3f, 1.0f));
                                float x0 = canvas_pos.x + f * col_w;
                                draw->AddRectFilled(
                                    ImVec2(x0, canvas_pos.y),
                                    ImVec2(x0 + col_w,
                                           canvas_pos.y + canvas_size.y),
                                    color);
                            }
                            draw->AddRect(canvas_pos,
                                ImVec2(canvas_pos.x + canvas_size.x,
                                       canvas_pos.y + canvas_size.y),
                                ImGui::GetColorU32(ImGuiCol_Border));
                            ImGui::TreePop();
                        }

                        // Motor output.
                        if (ImGui::TreeNodeEx("Motor Output",
                                ImGuiTreeNodeFlags_DefaultOpen |
                                ImGuiTreeNodeFlags_SpanAvailWidth)) {
                            auto mcmd = brain.GetMotorCommand();
                            ImGui::Text("Forward:  %+.1f mm/s",
                                        mcmd.forward_velocity);
                            ImGui::Text("Angular:  %+.2f rad/s",
                                        mcmd.angular_velocity);
                            float drive = std::clamp(
                                mcmd.approach_drive, -5.0f, 5.0f);
                            float norm = (drive + 5.0f) / 10.0f;
                            ImGui::Text("Valence:  %+.2f",
                                        mcmd.approach_drive);
                            ImGui::ProgressBar(norm, ImVec2(-1, 0),
                                drive > 0.1f ? "Approach"
                                : drive < -0.1f ? "Avoid" : "Neutral");
                            if (mcmd.freeze > 0.5f) {
                                ImGui::TextColored(
                                    ImVec4(1, 0.5f, 0.2f, 1), "FREEZE");
                            }
                            ImGui::TreePop();
                        }

                        // Feature toggles.
                        ImGui::Separator();
                        auto& features = brain.Features();
                        ImGui::Checkbox("STDP Learning", &features.stdp);
                        ImGui::Checkbox("Homeostasis", &features.homeostasis);
                        ImGui::Checkbox("Gap Junctions", &features.gap_junctions);
                        ImGui::Checkbox("Neuromodulation", &features.neuromodulation);
                        ImGui::Checkbox("SFA (Adaptation)", &features.sfa);
                        ImGui::Checkbox("Conduction Delays", &features.conduction_delays);
                        ImGui::Checkbox("Short-Term Plasticity",
                                        &features.short_term_plasticity);
                        ImGui::Checkbox("Structural Plasticity",
                                        &features.structural_plasticity);
                        ImGui::Checkbox("Inhibitory Plasticity",
                                        &features.inhibitory_plasticity);
                        ImGui::Checkbox("Neuromod Effects",
                                        &features.neuromodulator_effects);

                        ImGui::PopID();
                    }
#endif

                    // -- Controls --
                    if (ImGui::CollapsingHeader("Controls",
                            ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::Text("Fly");
                        ImGui::TextDisabled(
                            "  W/A/S/D: walk / turn / stop");
                        ImGui::TextDisabled(
                            "  Space: run  |  Shift: fly");
                        ImGui::Spacing();
                        ImGui::Text("Body Camera");
                        ImGui::TextDisabled("  Orbit: left drag");
                        ImGui::TextDisabled("  Pan: right drag");
                        ImGui::TextDisabled("  Zoom: scroll");
                        ImGui::Spacing();
                        ImGui::Text("Brain Camera");
                        ImGui::TextDisabled("  Orbit/Pan/Zoom in brain panel");
                        ImGui::TextDisabled("  R: reset view");
                    }

                    ImGui::End();
                    ImGui::PopStyleVar();
                }

                // -- Brain panel (bottom-right) --
                if (brain_vp.initialized && brain_vp.Texture()) {
                    constexpr float kBrainW = 500.0f, kBrainH = 400.0f;
                    ImGui::SetNextWindowPos(
                        ImVec2(static_cast<float>(win_w) - kBrainW - 8.0f,
                               static_cast<float>(win_h) - kBrainH - 8.0f),
                        ImGuiCond_Always);
                    ImGui::SetNextWindowSize(
                        ImVec2(kBrainW, kBrainH), ImGuiCond_Always);
                    ImGui::PushStyleVar(
                        ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
                    if (ImGui::Begin("Brain##viewport", nullptr,
                            ImGuiWindowFlags_NoScrollbar |
                            ImGuiWindowFlags_NoScrollWithMouse |
                            ImGuiWindowFlags_NoMove)) {
                        ImVec2 avail = ImGui::GetContentRegionAvail();
                        ImVec2 img_pos = ImGui::GetCursorScreenPos();
                        ImGui::Image(
                            (ImTextureID)(uintptr_t)brain_vp.Texture(),
                            avail, ImVec2(0, 1), ImVec2(1, 0));

                        // Mouse interaction on brain image.
                        if (ImGui::IsItemHovered()) {
                            ImGuiIO& io = ImGui::GetIO();
                            if (io.MouseWheel != 0.0f) {
                                brain_vp.camera.radius *=
                                    (io.MouseWheel > 0) ? 0.9f : 1.1f;
                                brain_vp.camera.radius = std::clamp(
                                    brain_vp.camera.radius, 50.0f, 2000.0f);
                            }
                            if (ImGui::IsKeyPressed(ImGuiKey_R))
                                brain_vp.camera.Reset();
                            if (ImGui::IsMouseDragging(
                                    ImGuiMouseButton_Left)) {
                                ImVec2 delta = io.MouseDelta;
                                brain_vp.camera.azimuth += delta.x * 0.005f;
                                brain_vp.camera.elevation += delta.y * 0.005f;
                                brain_vp.camera.elevation = std::clamp(
                                    brain_vp.camera.elevation, -1.5f, 1.5f);
                            }
                            if (ImGui::IsMouseDragging(
                                    ImGuiMouseButton_Right) ||
                                ImGui::IsMouseDragging(
                                    ImGuiMouseButton_Middle)) {
                                ImVec2 delta = io.MouseDelta;
                                float pan_speed =
                                    brain_vp.camera.radius * 0.002f;
                                brain_vp.camera.target_x -=
                                    delta.x * pan_speed *
                                    std::cos(brain_vp.camera.azimuth);
                                brain_vp.camera.target_z +=
                                    delta.x * pan_speed *
                                    std::sin(brain_vp.camera.azimuth);
                                brain_vp.camera.target_y +=
                                    delta.y * pan_speed;
                            }
                        }

                        // Transparent Regions overlay (top-left of brain image).
                        ImGui::SetCursorScreenPos(ImVec2(img_pos.x + 4, img_pos.y + 4));
                        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 2));
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.15f, 0.7f));
                        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.2f, 0.3f, 0.8f));
                        if (ImGui::SmallButton(regions_open ? "Regions [-]" : "Regions [+]"))
                            regions_open = !regions_open;
                        ImGui::PopStyleColor(2);
                        ImGui::PopStyleVar();

                        if (regions_open) {
                            ImGui::SetCursorScreenPos(ImVec2(img_pos.x + 4, img_pos.y + 26));
                            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.12f, 0.85f));
                            float reg_h = 26.0f + 8 * 22.0f + 12.0f;
                            ImGui::BeginChild("##regions_overlay", ImVec2(300, reg_h), true);

                            float tb = (ImGui::GetContentRegionAvail().x - 8) / 3.0f;
                            if (ImGui::Button("All", ImVec2(tb, 0))) {
                                for (int i = 0; i < 14; ++i) region_size[i] = 4.0f;
                                region_dirty = true;
                            }
                            ImGui::SameLine();
                            if (ImGui::Button("None", ImVec2(tb, 0))) {
                                for (int i = 0; i < 14; ++i) region_size[i] = 0.0f;
                                region_dirty = true;
                            }
                            ImGui::SameLine();
                            if (ImGui::Button("Reset", ImVec2(tb, 0))) {
                                const float defaults[14] = {1,2,2,4,4,4,4,4,4,4,4,4,4,4};
                                for (int i = 0; i < 14; ++i) region_size[i] = defaults[i];
                                region_dirty = true;
                            }

                            const ImGuiTableFlags tflags =
                                ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_NoPadOuterX;

                            if (ImGui::BeginTable("##reg", 3, tflags)) {
                                ImGui::TableSetupColumn("##vis",  ImGuiTableColumnFlags_WidthFixed, 20.0f);
                                ImGui::TableSetupColumn("##name", ImGuiTableColumnFlags_WidthStretch, 1.0f);
                                ImGui::TableSetupColumn("##size", ImGuiTableColumnFlags_WidthStretch, 0.7f);

                                auto eye_swatch = [&](int idx) {
                                    ImGui::TableNextColumn();
                                    float cr, cg, cb;
                                    nmfly::RegionColor(idx, cr, cg, cb);
                                    bool visible = region_size[idx] > 0.0f;
                                    float alpha = visible ? 1.0f : 0.25f;
                                    if (ImGui::ColorButton(("##e" + std::to_string(idx)).c_str(),
                                                           ImVec4(cr, cg, cb, alpha),
                                                           ImGuiColorEditFlags_NoTooltip, ImVec2(14, 14))) {
                                        region_size[idx] = visible ? 0.0f : 4.0f;
                                        region_dirty = true;
                                    }
                                    if (ImGui::IsItemHovered())
                                        ImGui::SetTooltip(visible ? "Hide" : "Show");
                                };

                                auto leaf = [&](const char* label, int idx) {
                                    ImGui::TableNextRow();
                                    eye_swatch(idx);
                                    ImGui::TableNextColumn();
                                    ImGui::AlignTextToFramePadding();
                                    ImGui::Text("%s", label);
                                    ImGui::TableNextColumn();
                                    ImGui::SetNextItemWidth(-1);
                                    if (ImGui::SliderFloat(("##s" + std::to_string(idx)).c_str(),
                                                           &region_size[idx], 0.0f, 12.0f, "%.0f"))
                                        region_dirty = true;
                                };

                                auto group_slider = [&](const char* id, const int* indices, int n) {
                                    ImGui::TableNextColumn();
                                    float sum = 0;
                                    for (int i = 0; i < n; ++i) sum += region_size[indices[i]];
                                    float avg = sum / static_cast<float>(n);
                                    ImGui::SetNextItemWidth(-1);
                                    if (ImGui::SliderFloat(id, &avg, 0.0f, 12.0f, "%.0f")) {
                                        for (int i = 0; i < n; ++i) region_size[indices[i]] = avg;
                                        region_dirty = true;
                                    }
                                };

                                auto group_eye = [&](const char* id, int color_idx,
                                                     const int* indices, int n) {
                                    ImGui::TableNextColumn();
                                    float cr, cg, cb;
                                    nmfly::RegionColor(color_idx, cr, cg, cb);
                                    bool any_vis = false;
                                    for (int i = 0; i < n; ++i)
                                        if (region_size[indices[i]] > 0.0f) any_vis = true;
                                    float alpha = any_vis ? 1.0f : 0.25f;
                                    if (ImGui::ColorButton(id, ImVec4(cr, cg, cb, alpha),
                                                           ImGuiColorEditFlags_NoTooltip, ImVec2(14, 14))) {
                                        float val = any_vis ? 0.0f : 4.0f;
                                        for (int i = 0; i < n; ++i) region_size[indices[i]] = val;
                                        region_dirty = true;
                                    }
                                };

                                auto pair = [&](const char* name, int l, int r) {
                                    int idx[2] = {l, r};
                                    ImGui::TableNextRow();
                                    group_eye(("##ge" + std::to_string(l)).c_str(), l, idx, 2);
                                    ImGui::TableNextColumn();
                                    bool open = ImGui::TreeNodeEx(name, ImGuiTreeNodeFlags_SpanAvailWidth);
                                    group_slider(("##a" + std::to_string(l)).c_str(), idx, 2);
                                    if (open) {
                                        leaf("L", l);
                                        leaf("R", r);
                                        ImGui::TreePop();
                                    }
                                };

                                leaf("Protocerebrum", 0);
                                pair("Optic Lobe", 1, 2);
                                {
                                    static const int mb[] = {3, 4, 5, 6};
                                    ImGui::TableNextRow();
                                    group_eye("##ge_mb", 3, mb, 4);
                                    ImGui::TableNextColumn();
                                    bool open = ImGui::TreeNodeEx("Mushroom Body",
                                        ImGuiTreeNodeFlags_SpanAvailWidth);
                                    group_slider("##cat_mb", mb, 4);
                                    if (open) {
                                        pair("Calyx", 3, 4);
                                        pair("Lobe", 5, 6);
                                        ImGui::TreePop();
                                    }
                                }
                                pair("Antennal Lobe", 7, 8);
                                leaf("Central Complex", 9);
                                pair("Lateral Horn", 10, 11);
                                leaf("SEZ", 12);

                                ImGui::EndTable();
                            }

                            ImGui::EndChild();
                            ImGui::PopStyleColor();
                        }
                    }
                    ImGui::End();
                    ImGui::PopStyleVar();

                    // Apply region size changes to brain viewport.
                    if (region_dirty) {
                        region_dirty = false;
                        if (brain_window) {
                            glfwMakeContextCurrent(brain_window);
                            brain_vp.SetRegionSizes(region_size);
                            glfwMakeContextCurrent(body_window);
                        }
                    }
                }

                // Render ImGui over MuJoCo.
                ImGui::Render();
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
                glViewport(0, 0, fb_w, fb_h);
                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

                // Body camera mouse controls (when ImGui not capturing).
                if (!ImGui::GetIO().WantCaptureMouse) {
                    double mx, my;
                    glfwGetCursorPos(body_window, &mx, &my);
                    bool left = glfwGetMouseButton(
                        body_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
                    bool right = glfwGetMouseButton(
                        body_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
                    bool mid = glfwGetMouseButton(
                        body_window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;

                    if (left && !body_mouse_left) {
                        body_mouse_left = true;
                        body_last_mx = mx; body_last_my = my;
                    }
                    if (!left) body_mouse_left = false;
                    if (right && !body_mouse_right) {
                        body_mouse_right = true;
                        body_last_mx = mx; body_last_my = my;
                    }
                    if (!right) body_mouse_right = false;
                    if (mid && !body_mouse_mid) {
                        body_mouse_mid = true;
                        body_last_mx = mx; body_last_my = my;
                    }
                    if (!mid) body_mouse_mid = false;

                    double dx = mx - body_last_mx;
                    double dy = my - body_last_my;
                    if (body_mouse_left)
                        sim.RotateCamera(dx, dy, win_w, win_h);
                    else if (body_mouse_right)
                        sim.TranslateCamera(dx, dy, win_w, win_h);
                    else if (body_mouse_mid)
                        sim.ZoomCamera(dy);
                    body_last_mx = mx;
                    body_last_my = my;
                }

                glfwSwapBuffers(body_window);
            }
#endif

        }
    }

    auto wall_end = Clock::now();
    double wall_sec = std::chrono::duration<double>(wall_end - wall_start).count();
    double sim_sec = sim.sim_time();

    printf("[flygame] Done. %d steps, %.1fs sim time, %.1fs wall time\n",
           total_steps, sim_sec, wall_sec);
    printf("[flygame] Speed: %.1fx realtime, %.0f steps/sec\n",
           sim_sec / std::max(wall_sec, 0.001), total_steps / std::max(wall_sec, 0.001));

    // Test mode: verify behavior at each phase.
    // Indices: 0=start, 1=end-idle, 2=end-W, 3=end-W+A, 4=end-W+D, 5=end-idle2
    if (cfg.test_mode) {
        int pass = 0, fail = 0;

        auto check = [&](const char* name, bool ok, const char* detail) {
            printf("[test] %s: %s (%s)\n", name, ok ? "PASS" : "FAIL", detail);
            if (ok) pass++; else fail++;
        };

        char buf[128];

        // Phase 0-1 (idle 0-0.5s): fly should barely move from start position.
        float idle_dx = test_pos[1][0] - test_pos[0][0];
        float idle_dy = test_pos[1][1] - test_pos[0][1];
        float idle_dist = std::sqrt(idle_dx*idle_dx + idle_dy*idle_dy);
        snprintf(buf, sizeof(buf), "drift=%.4f mm", idle_dist * 1000.0f);
        check("idle_stay_still", idle_dist < 0.5f, buf);  // < 500mm settling OK

        // Phase 1-2 (W 0.5-1.5s): fly should move forward.
        float fwd_dx = test_pos[2][0] - test_pos[1][0];
        float fwd_dy = test_pos[2][1] - test_pos[1][1];
        float fwd_dist = std::sqrt(fwd_dx*fwd_dx + fwd_dy*fwd_dy);
        snprintf(buf, sizeof(buf), "dist=%.4f", fwd_dist);
        check("forward_moves", fwd_dist > 0.001f, buf);

        // Phase 2-3 (W+A 1.5-2.0s): heading should change.
        // A = left turn = positive heading (counterclockwise in MuJoCo).
        float turn_a = test_pos[3][2] - test_pos[2][2];
        while (turn_a > nmfly::kPi) turn_a -= nmfly::kTwoPi;
        while (turn_a < -nmfly::kPi) turn_a += nmfly::kTwoPi;
        snprintf(buf, sizeof(buf), "dheading=%.4f rad", turn_a);
        check("turn_left_a", turn_a > 0.01f, buf);

        // Phase 3-4 (W+D 2.0-2.5s): heading should change.
        // D = right turn = negative heading (clockwise in MuJoCo).
        float turn_d = test_pos[4][2] - test_pos[3][2];
        while (turn_d > nmfly::kPi) turn_d -= nmfly::kTwoPi;
        while (turn_d < -nmfly::kPi) turn_d += nmfly::kTwoPi;
        snprintf(buf, sizeof(buf), "dheading=%.4f rad", turn_d);
        check("turn_right_d", turn_d < -0.01f, buf);

        // Phase 4-5 (idle 2.5-3.0s): should move less than the walking phase.
        float stop_dx = test_pos[5][0] - test_pos[4][0];
        float stop_dy = test_pos[5][1] - test_pos[4][1];
        float stop_dist = std::sqrt(stop_dx*stop_dx + stop_dy*stop_dy);
        snprintf(buf, sizeof(buf), "stop=%.4f vs fwd=%.4f", stop_dist, fwd_dist);
        check("idle_slows_down", stop_dist < fwd_dist, buf);

        printf("[test] Results: %d/%d passed\n", pass, pass + fail);
        if (fail > 0) {
            brain_running = false;
            if (brain_thread.joinable()) brain_thread.join();
            sim.Shutdown();
            return 1;
        }
    }

    // Stop brain thread.
    brain_running = false;
    if (brain_thread.joinable()) brain_thread.join();

#ifdef NMFLY_VIEWER
    if (body_window) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        if (brain_window) {
            brain_vp.Shutdown();
            glfwDestroyWindow(brain_window);
        }
    }
#endif

    sim.Shutdown();
    return 0;
}
