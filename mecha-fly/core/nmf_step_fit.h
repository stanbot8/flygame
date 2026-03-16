#pragma once
// Parameter fitting harness for NmfStepper.
// Uses fwmc::ExperimentOptimizer to find stepper parameters that best
// reproduce the recorded NMF walk trajectories.
// Tracks both knee (femur-tibia junction) and foot tip positions.

#include "nmf_stepper.h"
#include "optimizer/core/optimizer.h"

namespace nmfly {

struct StepFitResult {
    double lambda, stride_scale, step_height, swing_apex_fwd, adhesion_ramp_width;
    double fitness;
};

// Reference positions from walk data: knee + foot for each leg and sample.
struct RefPositions {
    Vec3 knee[6][kWalkSamples];
    Vec3 foot[6][kWalkSamples];
};

inline void ComputeReferencePositions(RefPositions& ref) {
    for (int leg = 0; leg < 6; ++leg) {
        for (int s = 0; s < kWalkSamples; ++s) {
            double phase = s * kTwoPi / (kWalkSamples - 1);
            double angles[7];
            for (int j = 0; j < 7; ++j)
                angles[j] = WalkLerp(leg, j, phase);
            FkChain chain = FlyFKChain(leg, angles);
            ref.knee[leg][s] = chain.tibia_pos;  // "knee" = tibia joint
            ref.foot[leg][s] = chain.foot_pos;
        }
    }
}

// Backward compat: old signature for test_fk_ik.cc
inline void ComputeReferencePositions(Vec3 ref_pos[6][kWalkSamples]) {
    for (int leg = 0; leg < 6; ++leg)
        for (int s = 0; s < kWalkSamples; ++s) {
            double phase = s * kTwoPi / (kWalkSamples - 1);
            double angles[7];
            for (int j = 0; j < 7; ++j)
                angles[j] = WalkLerp(leg, j, phase);
            ref_pos[leg][s] = FlyFK(leg, angles).foot_pos;
        }
}

inline StepFitResult FitStepperParams(fwmc::LogFn log_fn = nullptr) {
    RefPositions ref;
    ComputeReferencePositions(ref);

    fwmc::ExperimentOptimizer opt;
    opt.axes = {
        {"lambda",              0.001f, 1.0f},
        {"stride_scale",        0.5f,   3.0f},
        {"step_height",         0.01f,  0.5f},
        {"swing_apex_fwd",      0.1f,   0.5f},
        {"adhesion_ramp_width", 0.05f,  0.6f},
    };
    opt.n_iterations = 1000;
    opt.n_seeds = 1;
    opt.log_fn = log_fn;

    auto best = opt.Run([&](const std::vector<float>& params, uint32_t) -> float {
        NmfStepper stepper;
        stepper.lambda = params[0];
        stepper.stride_scale = params[1];
        stepper.step_height = params[2];
        stepper.swing_apex_fwd = params[3];
        stepper.adhesion_ramp_width = params[4];
        stepper.Init(12.0);
        stepper.SetDrive(1.0, 0.0);

        double dt = 0.0001;
        int steps_per_cycle = static_cast<int>(1.0 / (12.0 * dt));
        int sample_interval = steps_per_cycle / kWalkSamples;

        double rms_sum = 0.0;
        int n = 0;
        for (int i = 0; i < steps_per_cycle; ++i) {
            stepper.Step(dt);

            if (i % sample_interval == 0 && n < kWalkSamples) {
                double ctrl[48];
                stepper.GetCtrl(ctrl);
                for (int leg = 0; leg < 6; ++leg) {
                    double angles[7];
                    for (int j = 0; j < 7; ++j)
                        angles[j] = ctrl[leg * 7 + j];
                    FkChain chain = FlyFKChain(leg, angles);
                    // Knee error (weighted same as foot).
                    Vec3 kd = chain.tibia_pos - ref.knee[leg][n];
                    rms_sum += kd.dot(kd);
                    // Foot error.
                    Vec3 fd = chain.foot_pos - ref.foot[leg][n];
                    rms_sum += fd.dot(fd);
                }
                n++;
            }
        }

        double rms = std::sqrt(rms_sum / (n * 12));  // 6 legs * 2 points
        return static_cast<float>(-rms);
    });

    return {
        static_cast<double>(best.params[0]),
        static_cast<double>(best.params[1]),
        static_cast<double>(best.params[2]),
        static_cast<double>(best.params[3]),
        static_cast<double>(best.params[4]),
        best.fitness
    };
}

}  // namespace nmfly
