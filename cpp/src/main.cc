// nmfly-sim: High-performance insect body simulation.
//
// Runs a MuJoCo fly body with tripod gait CPG, driven by motor commands
// from a brain simulator (FWMC or any controller speaking the TCP protocol).
//
// Usage:
//   nmfly-sim                          # standalone, walk forward
//   nmfly-sim --brain 127.0.0.1:9100   # connect to FWMC brain
//   nmfly-sim --model fly.xml          # custom MJCF model
//   nmfly-sim --headless               # no viewer, max speed

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <string>
#include <charconv>

#include "types.h"
#include "gait.h"
#include "nmf_cpg.h"
#include "proprio.h"
#include "sim.h"
#include "tcp.h"
#include "fly_model.h"

using namespace nmfly;
using Clock = std::chrono::steady_clock;

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
};

static void PrintUsage(const char* prog) {
    printf("nmfly-sim: High-performance insect body simulation\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --brain HOST:PORT   Connect to brain sim (default: standalone)\n");
    printf("  --model PATH        MJCF model file (default: flygym NMF)\n");
    printf("  --headless          No viewer window (max speed)\n");
    printf("  --dt SECONDS        Physics timestep (default: 0.0002)\n");
    printf("  --substeps N        Physics sub-steps per controller step (default: 5)\n");
    printf("  --duration SECONDS  Run for N seconds sim time (default: forever)\n");
    printf("  --speed MM_S        Standalone forward speed (default: 15)\n");
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

    printf("[nmfly-sim] Starting (dt=%.5f, substeps=%d, %s)\n",
           cfg.dt, cfg.substeps,
           cfg.standalone ? "standalone" : "brain-connected");

    // Set up simulation.
    SimConfig sim_cfg;
    sim_cfg.dt         = cfg.dt;
    sim_cfg.render     = !cfg.headless;
    sim_cfg.substeps   = cfg.substeps;
    sim_cfg.sync_every = 10;
    sim_cfg.model_path = cfg.model_path;

    Sim sim;
    std::string xml;
    if (cfg.model_path.empty()) {
        // Look for the pre-exported NMF model (with actuators) next to the exe.
        namespace fs = std::filesystem;
        fs::path exe_dir = fs::path(argv[0]).parent_path();
        fs::path search[] = {
            exe_dir / "data" / "nmf_complete.xml",
            exe_dir / ".." / "data" / "nmf_complete.xml",
            exe_dir / ".." / ".." / "cpp" / "data" / "nmf_complete.xml",
        };
        bool found = false;
        for (auto& p : search) {
            if (fs::exists(p)) {
                cfg.model_path = fs::canonical(p).string();
                sim_cfg.model_path = cfg.model_path;
                found = true;
                break;
            }
        }
        if (found) {
            printf("[nmfly-sim] Using NMF model: %s\n",
                   cfg.model_path.c_str());
        } else {
            xml = StickFlyMJCF();
            printf("[nmfly-sim] NMF model not found, using built-in stick fly\n");
        }
    }

    if (!sim.Setup(sim_cfg, xml)) {
        fprintf(stderr, "[nmfly-sim] Failed to set up simulation\n");
        return 1;
    }
    printf("[nmfly-sim] MuJoCo ready (%d actuators, %d DOF)\n",
           sim.model()->nu, sim.model()->nv);

    // NMF initialization: set qpos to tripod pose, then settle with
    // actuators holding that pose (no qpos tracking drift).
    int nu = sim.model()->nu;
    NmfCpg nmf_cpg;
    if (nu == 48) {
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
        printf("[nmfly-sim] NMF init (tripod pose): z=%.4f, ncon=%d\n",
               sim.data()->qpos[2], sim.data()->ncon);

        // Warmup: hold tripod pose via actuators while contacts settle.
        // Do NOT track qpos (that lets gravity collapse the pose).
        for (int s = 0; s < 500; ++s) {
            // ctrl stays constant: actuators actively hold the pose.
            mj_step(sim.model(), sim.data());
            if (s < 3 || s == 499)
                printf("[nmfly-sim]   warmup %d: z=%.4f, ncon=%d\n",
                       s, sim.data()->qpos[2], sim.data()->ncon);
            if (s % 100 == 0) sim.SyncViewerPublic();
        }
        sim.SyncViewerPublic();
        printf("[nmfly-sim] NMF settled (z=%.4f, ncon=%d)\n",
               sim.data()->qpos[2], sim.data()->ncon);

        // Start walking in standalone mode.
        if (cfg.standalone)
            nmf_cpg.SetDrive(1.0, 0.0);
    }

    // Connect to brain if requested.
    TcpClient tcp;
    if (!cfg.standalone) {
        printf("[nmfly-sim] Connecting to brain at %s:%d...\n",
               cfg.brain_host.c_str(), cfg.brain_port);
        if (tcp.Connect(cfg.brain_host, cfg.brain_port)) {
            printf("[nmfly-sim] Connected to brain\n");
        } else {
            printf("[nmfly-sim] Could not connect. Running standalone.\n");
            cfg.standalone = true;
        }
    }

    // Gait state.
    GaitState gait;
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

    printf("[nmfly-sim] Running%s...\n",
           cfg.headless ? " (headless)" : " (with viewer)");

    // Main loop.
    while (sim.IsRunning()) {
        // Duration limit.
        if (cfg.duration > 0.0f && sim.sim_time() >= cfg.duration)
            break;

        // TCP exchange at reduced rate.
        if (!cfg.standalone && tcp.IsConnected() &&
            ctrl_count % exchange_every == 0) {
            BodyState state = sim.ReadState();
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
        if (nu == 48) {
            // NeuroMechFly: Kuramoto CPG with real trajectory lookup.
            // Map smoothed motor command to CPG drive (magnitude + turn).
            double magnitude = std::clamp(
                static_cast<double>(smooth_cmd.forward_velocity) / 15.0, 0.0, 1.5);
            double turn = std::clamp(
                static_cast<double>(smooth_cmd.angular_velocity) / 3.0, -1.0, 1.0);
            nmf_cpg.SetDrive(magnitude, turn);

            // Integrate CPG per physics substep (not per controller step)
            // for smooth phase evolution matching flygym's integration rate.
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
                printf("[nmfly-sim] Physics reset (instability)\n");
                ctrl_count = 0;
                continue;
            }

            sim.add_steps(cfg.substeps);
            total_steps += cfg.substeps;
            ctrl_count++;

            if (ctrl_count <= 5 || ctrl_count % 500 == 0)
                printf("[nmfly-sim] frame %d: z=%.4f ncon=%d\n",
                       ctrl_count, sim.data()->qpos[2], sim.data()->ncon);

            if (!cfg.headless)
                sim.SyncViewerPublic();

        } else {
            // Stick fly: synthetic gait.
            float step_dt = cfg.dt * cfg.substeps;
            gait.Update(smooth_cmd, step_dt);
            float targets[kLegs][kJointsPerLeg];
            ComputeJointTargets(gait, targets);

            // Stumble correction.
            BodyState pre_state = sim.ReadState();
            for (int leg = 0; leg < kLegs; ++leg) {
                if (IsSwingPhase(gait, leg) && pre_state.contacts[leg] > 0.5f)
                    gait.TriggerStumble(leg);
            }
            // Stick fly or other model: 4 joints/leg (yaw, pitch, CTr, FTi)
            float ctrl[64] = {};
            int idx = 0;
            for (int leg = 0; leg < kLegs && idx < nu; ++leg) {
                ctrl[idx++] = targets[leg][0];
                ctrl[idx++] = targets[leg][1];
                ctrl[idx++] = targets[leg][3];
                ctrl[idx++] = targets[leg][5];
            }
            if (!sim.Step(ctrl, nu)) {
                printf("[nmfly-sim] Physics reset (instability)\n");
                gait = GaitState();
                cmd = cfg.standalone ? MotorCommand{cfg.fwd_speed} : MotorCommand();
                ctrl_count = 0;
                continue;
            }
            total_steps += cfg.substeps;
            ctrl_count++;
        }
    }

    auto wall_end = Clock::now();
    double wall_sec = std::chrono::duration<double>(wall_end - wall_start).count();
    double sim_sec = sim.sim_time();

    printf("[nmfly-sim] Done. %d steps, %.1fs sim time, %.1fs wall time\n",
           total_steps, sim_sec, wall_sec);
    printf("[nmfly-sim] Speed: %.1fx realtime, %.0f steps/sec\n",
           sim_sec / std::max(wall_sec, 0.001), total_steps / std::max(wall_sec, 0.001));

    sim.Shutdown();
    return 0;
}
