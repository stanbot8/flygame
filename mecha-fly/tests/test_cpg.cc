// Tests for NeuroMechFly Kuramoto CPG controller.
// Validates phase locking, tripod gait coordination, turn dynamics,
// adhesion timing, and joint angle interpolation.

#include <cmath>
#include "nmf_cpg.h"
#include "test_harness.h"

constexpr double kTol = 1e-4;

// Helper: wrap phase to [0, 2pi).
static double wrap(double p) {
  return p - kTwoPi * std::floor(p / kTwoPi);
}

// Helper: angular distance (unsigned, [0, pi]).
static double angle_diff(double a, double b) {
  double d = std::fmod(std::abs(a - b), kTwoPi);
  return d > kPi ? kTwoPi - d : d;
}

TEST(cpg_init_phases) {
  NmfCpg cpg;
  cpg.Init(12.0, 1.0);

  // Tripod A (LF, LH, RM) start at 0
  CHECK(std::abs(cpg.phase[0]) < kTol);  // LF
  CHECK(std::abs(cpg.phase[2]) < kTol);  // LH
  CHECK(std::abs(cpg.phase[4]) < kTol);  // RM

  // Tripod B (LM, RF, RH) start at pi
  CHECK(std::abs(cpg.phase[1] - kPi) < kTol);  // LM
  CHECK(std::abs(cpg.phase[3] - kPi) < kTol);  // RF
  CHECK(std::abs(cpg.phase[5] - kPi) < kTol);  // RH

  // Amplitude starts at 0 (ramp up from zero)
  for (int i = 0; i < 6; ++i)
    CHECK(std::abs(cpg.amplitude[i]) < kTol);

  // Target amplitude = 1.0
  for (int i = 0; i < 6; ++i)
    CHECK(std::abs(cpg.target_amp[i] - 1.0) < kTol);
}

TEST(cpg_phase_advance) {
  NmfCpg cpg;
  cpg.Init(10.0, 1.0);

  // After one full period (0.1s at 10Hz), phases should advance ~2pi
  double total_dt = 0.0;
  int steps = 10000;
  double dt = 1.0 / (10.0 * steps);  // 0.1s total

  for (int i = 0; i < steps; ++i) {
    cpg.Step(dt);
    total_dt += dt;
  }

  // LF phase should have advanced ~2pi (modulo coupling)
  double p = wrap(cpg.phase[0]);
  // With coupling, the phase might drift slightly, but should be close to 0
  CHECK(p < 0.5 || p > kTwoPi - 0.5);
}

TEST(cpg_tripod_lock) {
  NmfCpg cpg;
  cpg.Init(12.0, 1.0);

  // Run for 1 second to let phases lock
  double dt = 0.001;
  for (int i = 0; i < 1000; ++i)
    cpg.Step(dt);

  // Tripod A legs should be in-phase (phase diff ~0)
  double lf = wrap(cpg.phase[0]);
  double lh = wrap(cpg.phase[2]);
  double rm = wrap(cpg.phase[4]);
  CHECK(angle_diff(lf, lh) < 0.3);
  CHECK(angle_diff(lf, rm) < 0.3);

  // Tripod B legs should be in-phase
  double lm = wrap(cpg.phase[1]);
  double rf = wrap(cpg.phase[3]);
  double rh = wrap(cpg.phase[5]);
  CHECK(angle_diff(lm, rf) < 0.3);
  CHECK(angle_diff(lm, rh) < 0.3);

  // A vs B should be anti-phase (diff ~pi)
  CHECK(angle_diff(lf, lm) > 2.5);
}

TEST(cpg_amplitude_convergence) {
  NmfCpg cpg;
  cpg.Init(12.0, 1.0);

  // After 0.5s, amplitudes should be near target (alpha=20, tau=0.05s)
  for (int i = 0; i < 5000; ++i)
    cpg.Step(0.0001);

  for (int leg = 0; leg < 6; ++leg)
    CHECK(std::abs(cpg.amplitude[leg] - 1.0) < 0.05);
}

TEST(cpg_set_drive_turn) {
  NmfCpg cpg;
  cpg.Init(12.0, 1.0);

  // Turn right: left legs should have higher amplitude
  cpg.SetDrive(1.0, 1.0);  // full right turn

  // Left legs (0,1,2): magnitude + turn*0.8 = 1.0 + 0.8 = 1.8 -> clamped to 1.5
  CHECK(std::abs(cpg.target_amp[0] - 1.5) < kTol);  // LF
  CHECK(std::abs(cpg.target_amp[1] - 1.5) < kTol);  // LM
  CHECK(std::abs(cpg.target_amp[2] - 1.5) < kTol);  // LH

  // Right legs (3,4,5): magnitude - turn*0.8 = 1.0 - 0.8 = 0.2
  CHECK(std::abs(cpg.target_amp[3] - 0.2) < kTol);  // RF
  CHECK(std::abs(cpg.target_amp[4] - 0.2) < kTol);  // RM
  CHECK(std::abs(cpg.target_amp[5] - 0.2) < kTol);  // RH
}

TEST(cpg_set_drive_stop) {
  NmfCpg cpg;
  cpg.Init(12.0, 1.0);

  // Zero magnitude should stop all legs
  cpg.SetDrive(0.0, 0.0);
  for (int i = 0; i < 6; ++i)
    CHECK(std::abs(cpg.target_amp[i]) < kTol);
}

TEST(cpg_ctrl_output) {
  NmfCpg cpg;
  cpg.Init(12.0, 1.0);

  // Let amplitude converge
  for (int i = 0; i < 10000; ++i)
    cpg.Step(0.0001);

  double ctrl[48] = {};
  cpg.GetCtrl(ctrl);

  // Joint angles should be finite and reasonable (radians)
  for (int i = 0; i < 42; ++i) {
    CHECK(std::isfinite(ctrl[i]));
    CHECK(ctrl[i] > -kPi && ctrl[i] < kPi);
  }

  // Adhesion values should be in [0, 3]
  for (int i = 42; i < 48; ++i) {
    CHECK(ctrl[i] >= -0.01);
    CHECK(ctrl[i] <= 3.01);
  }
}

TEST(cpg_adhesion_stance_swing) {
  NmfCpg cpg;
  cpg.Init(12.0, 1.0);

  // Force amplitude=1 immediately
  for (int i = 0; i < 6; ++i)
    cpg.amplitude[i] = 1.0;

  // At phase=pi (mid-stance), adhesion should be high
  // At phase=pi/2 (mid-swing), adhesion should be low
  // We test by setting specific phases

  // Tripod A at mid-stance (phase = pi + small offset past ramp)
  cpg.phase[0] = kPi + 0.5;  // LF mid-stance
  cpg.phase[1] = 0.5;          // LM mid-swing

  double ctrl[48];
  cpg.GetCtrl(ctrl);

  // LF (leg 0) should have high adhesion (stance)
  CHECK(ctrl[42 + 0] > 2.0);

  // LM (leg 1) at phase ~0.5 (past ramp, into swing)
  CHECK(ctrl[42 + 1] < 0.5);
}

TEST(cpg_neutral_pose) {
  NmfCpg cpg;
  cpg.Init(12.0, 1.0);

  // With amplitude = 0, ctrl should equal neutral pose (trajectory at pi)
  for (int i = 0; i < 6; ++i)
    cpg.amplitude[i] = 0.0;

  double ctrl1[48], ctrl2[48];
  cpg.phase[0] = 0.0;
  cpg.GetCtrl(ctrl1);
  cpg.phase[0] = 1.5;
  cpg.GetCtrl(ctrl2);

  // Joint angles for leg 0 should be identical regardless of phase
  for (int j = 0; j < 7; ++j)
    CHECK(std::abs(ctrl1[j] - ctrl2[j]) < kTol);
}

TEST(cpg_walk_lerp_continuity) {
  // Walk trajectory interpolation should be smooth (no big jumps)
  for (int leg = 0; leg < 6; ++leg) {
    for (int joint = 0; joint < 7; ++joint) {
      double prev = WalkLerp(leg, joint, 0.0);
      for (int step = 1; step <= 360; ++step) {
        double phase = step * kTwoPi / 360.0;
        double curr = WalkLerp(leg, joint, phase);
        double delta = std::abs(curr - prev);
        // Maximum step should be small (< 0.2 rad per degree)
        CHECK(delta < 0.3);
        prev = curr;
      }
    }
  }
}

int main() {
  return RunAllTests();
}
