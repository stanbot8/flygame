#pragma once
// Inverse kinematics for NeuroMechFly legs with walk-data prior regularization.
// Uses damped least-squares (Levenberg-Marquardt) on all 7 DOF.

#include "nmf_fk.h"

namespace nmfly {

struct IkConfig {
    double lambda = 0.1;    // regularization weight toward prior angles
    double damping = 1e-3;  // Levenberg-Marquardt damping factor
    int max_iter = 50;      // maximum iterations
    double tol = 1e-4;      // convergence tolerance (position error norm)
    double step_scale = 0.5; // step size scale (< 1 for stability)
};

struct IkResult {
    double angles[7];
    double position_error;  // residual ||foot - target||
    bool converged;
};

// Compute 7x3 Jacobian numerically (finite differences on FK).
// J[j][k] = d(foot_pos[k]) / d(angles[j])
inline void ComputeJacobian(int leg, const double angles[7], double J[7][3]) {
    constexpr double h = 1e-5;
    FkResult fk0 = FlyFK(leg, angles);

    for (int j = 0; j < 7; ++j) {
        double perturbed[7];
        for (int k = 0; k < 7; ++k) perturbed[k] = angles[k];
        perturbed[j] += h;

        FkResult fk1 = FlyFK(leg, perturbed);
        J[j][0] = (fk1.foot_pos.x - fk0.foot_pos.x) / h;
        J[j][1] = (fk1.foot_pos.y - fk0.foot_pos.y) / h;
        J[j][2] = (fk1.foot_pos.z - fk0.foot_pos.z) / h;
    }
}

// Solve 3x3 symmetric system Ax=b via Cramer's rule.
// A is stored as a[3][3], b as b[3], result in x[3].
// Returns false if det is near zero.
inline bool Solve3x3(const double A[3][3], const double b[3], double x[3]) {
    double det = A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1])
               - A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0])
               + A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]);
    if (std::abs(det) < 1e-20) return false;
    double inv_det = 1.0 / det;

    x[0] = (b[0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1])
          - A[0][1]*(b[1]*A[2][2] - A[1][2]*b[2])
          + A[0][2]*(b[1]*A[2][1] - A[1][1]*b[2])) * inv_det;
    x[1] = (A[0][0]*(b[1]*A[2][2] - A[1][2]*b[2])
          - b[0]*(A[1][0]*A[2][2] - A[1][2]*A[2][0])
          + A[0][2]*(A[1][0]*b[2] - b[1]*A[2][0])) * inv_det;
    x[2] = (A[0][0]*(A[1][1]*b[2] - b[1]*A[2][1])
          - A[0][1]*(A[1][0]*b[2] - b[1]*A[2][0])
          + b[0]*(A[1][0]*A[2][1] - A[1][1]*A[2][0])) * inv_det;
    return true;
}

// Inverse kinematics with walk-data prior regularization.
//
// Minimizes: ||foot_pos - target||^2 + lambda * ||angles - prior||^2
// Using damped pseudoinverse: delta = J^T (J J^T + mu*I)^-1 err - scale * lambda * (q - prior)
//
// angles[] uses actuator index order (same as FK, walk data, ctrl[]).
inline IkResult FlyIK(int leg, Vec3 target, const double prior[7],
                       const IkConfig& cfg = {}) {
    IkResult result;
    for (int j = 0; j < 7; ++j) result.angles[j] = prior[j];
    result.converged = false;

    for (int iter = 0; iter < cfg.max_iter; ++iter) {
        FkResult fk = FlyFK(leg, result.angles);
        Vec3 err = target - fk.foot_pos;
        result.position_error = err.norm();

        if (result.position_error < cfg.tol) {
            result.converged = true;
            return result;
        }

        // Compute Jacobian J[7][3]: J[joint][xyz]
        double J[7][3];
        ComputeJacobian(leg, result.angles, J);

        // Compute A = J J^T + mu*I  (3x3 matrix)
        double mu = cfg.damping;
        double A[3][3] = {};
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                double sum = 0;
                for (int j = 0; j < 7; ++j)
                    sum += J[j][r] * J[j][c];
                A[r][c] = sum + (r == c ? mu : 0.0);
            }
        }

        // Solve A * y = err  (3x1 vector)
        double e[3] = {err.x, err.y, err.z};
        double y[3];
        if (!Solve3x3(A, e, y)) break;  // singular

        // delta = J^T * y  (7x1)
        for (int j = 0; j < 7; ++j) {
            double dq = 0;
            for (int k = 0; k < 3; ++k)
                dq += J[j][k] * y[k];
            // Regularization toward prior.
            dq -= cfg.lambda * (result.angles[j] - prior[j]);
            result.angles[j] += cfg.step_scale * dq;
        }
    }

    // Final error check.
    FkResult fk = FlyFK(leg, result.angles);
    result.position_error = (target - fk.foot_pos).norm();
    result.converged = (result.position_error < cfg.tol);
    return result;
}

}  // namespace nmfly
