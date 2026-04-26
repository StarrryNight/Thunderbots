#pragma once

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <iostream>
#include <functional>
#include "software/logger/logger.h"
/**
 * Implementation of an unscented kalman filter.
 *
 * This is probably the best resource on the kalman filter for programmers:
 * https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
 *
 * This is a good overview if you are familiar with bayesian maths and state based control
 * theory:
 * https://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf
 *
 * @tparam DimX The dimension of the state
 * @tparam DimZ The dimension of measurement space
 */
template <int DimX, int DimZ>
class UnscentedKalmanFilter
{
   public:

    /**
     * Creates kalman filter with provided parameters.
     */
    UnscentedKalmanFilter(Eigen::Matrix<double, DimX, 1> x, Eigen::Matrix<double, DimX, DimX> P,
            Eigen::Matrix<double, DimX, DimX> Q,
            Eigen::Matrix<double, DimZ, DimX> H,
            Eigen::Matrix<double, DimZ, DimZ> R,
            std::function<Eigen::Matrix<double, 1, DimX>(Eigen::Matrix<double, 1, DimX>, double)> process_function,
            double alpha,
            double beta,
            double kappa);

    /**
     * Uses state model to innovate next state, taking into account expected behaviour
     * from control input.
     * @param input An extra input to pass to the process function, usually this is delta time
     */
    void predict(double input);

    /**
     * Incorporates measurement to state.
     * @param z Measurement
     */
    void update(Eigen::Matrix<double, DimZ, 1> z);

    Eigen::Matrix<double, DimX, 1> x;     // State
    Eigen::Matrix<double, DimX, DimX> P;  // State covariance
    Eigen::Matrix<double, DimX, DimX> Q;  // Process covariance
    Eigen::Matrix<double, DimZ, DimX> H;  // Converts state to measurement space
    Eigen::Matrix<double, DimZ, DimZ> R;  // Measurement noise covariance
    std::function<Eigen::Matrix<double, 1, DimX>(Eigen::Matrix<double, 1, DimX>, double)> process_function;  // Process model, takes ROW VECTORS

    

private:
    struct SigmaPoint {
        Eigen::Matrix<double, DimX, 1> position;
        int weight;
    };

    // PARAMETERS FOR THE VAN DER MERWE'S ALGORITHM FOR SIGMA POINT GENERATION
    // variable names are the same as in the literature
    // n is the dimension of x, 0 < alpha <= 1, beta = 2, kappa = 3 - n
    // hence total sigma point count will be 1 + n * DimX
    double alpha;
    double beta;
    double kappa;


    /**
     * Uses Van der Merwe's Scaled Sigma Point Algorithm to generate Sigma points and their weights to pass through the process function
     *
     * Sigma points are stored as a matrix where each row is a vector with the point
     * The weights are stored as a row vector, two weights are returned, the first for mean and the second for covariance
     */
    std::tuple<Eigen::Matrix<double, 2 * DimX + 1, DimX>, Eigen::Matrix<double, 1, 2 * DimX + 1>, Eigen::Matrix<double, 1, 2 * DimX + 1>> getSigmaPoints();

    std::pair<Eigen::Matrix<double, DimX, 1>, Eigen::Matrix<double, DimX, DimX>> unscented_transform(
        Eigen::Matrix<double, 1, 2 * DimX + 1> mean_weights, 
        Eigen::Matrix<double, 1, 2 * DimX + 1> covariance_weights, 
        Eigen::Matrix<double, 2 * DimX + 1, DimX> sigma_points,
        Eigen::Matrix<double, DimX, DimX> noise);
};

template <int DimX, int DimZ>
UnscentedKalmanFilter<DimX, DimZ>::UnscentedKalmanFilter(Eigen::Matrix<double, DimX, 1> x, Eigen::Matrix<double, DimX, DimX> P,
            Eigen::Matrix<double, DimX, DimX> Q,
            Eigen::Matrix<double, DimZ, DimX> H,
            Eigen::Matrix<double, DimZ, DimZ> R,
            std::function<Eigen::Matrix<double, 1, DimX>(Eigen::Matrix<double, 1, DimX>, double)> process_function,
            double alpha,
            double beta,
            double kappa) 
        : x(x),
          P(P),
          Q(Q),
          H(H),
          R(R),
          process_function(process_function),
          alpha(alpha),
          beta(beta),
          kappa(kappa)
    {
    }

template <int DimX, int DimZ>
void UnscentedKalmanFilter<DimX, DimZ>::predict(double input) {
    auto [sigma_points, mean_weights, covariance_weights] = getSigmaPoints();
    for (int i = 0; i <= 2 * DimX; i++) {
        sigma_points.row(i) = process_function(sigma_points.row(i), input);
    }
    auto result = unscented_transform(mean_weights, covariance_weights, sigma_points, Q);
    x = result.first;
    P = result.second;
}

template <int DimX, int DimZ>
void UnscentedKalmanFilter<DimX, DimZ>::update(Eigen::Matrix<double, DimZ, 1> z)
{
    Eigen::Matrix<double, DimZ, 1> y      = z - H * x;  // residual
    Eigen::Matrix<double, DimZ, DimZ> sum = H * P * H.transpose() + R;
    Eigen::Matrix<double, DimZ, DimZ> newSum =
        sum.unaryExpr([](double l) { return (fabs(l) < 1.0e-20) ? 0. : l; });
    Eigen::Matrix<double, DimX, DimZ> K =
        P * (H.transpose() *
             newSum.completeOrthogonalDecomposition().pseudoInverse());  // Kalman gain
    Eigen::Matrix<double, DimX, 1> newX = x + K * y;
    x                                   = newX;
    // Joseph equation is more stable than  P = (I-KH)P since the latter is
    // susceptible to floating point errors ruining symmetry
    Eigen::Matrix<double, DimX, DimX> posteriorCov =
        Eigen::Matrix<double, DimX, DimX>::Identity() - K * H;
    P = posteriorCov * P * posteriorCov.transpose() + K * R * K.transpose();
}

template <int DimX, int DimZ>
std::tuple<Eigen::Matrix<double, 2 * DimX + 1, DimX>, Eigen::Matrix<double, 1, 2 * DimX + 1>, Eigen::Matrix<double, 1, 2 * DimX + 1>> UnscentedKalmanFilter<DimX, DimZ>::getSigmaPoints() {
    double n = DimX;
    double lambda = alpha*alpha*(n + kappa) - n;
    
    Eigen::Matrix<double, 1, 2 * DimX + 1> covariance_weights = Eigen::Matrix<double, 1, 2 * DimX + 1>::Ones() / (2*(n + lambda)) ;
    Eigen::Matrix<double, 1, 2 * DimX + 1> mean_weights = Eigen::Matrix<double, 1, 2 * DimX + 1>::Ones();
    Eigen::Matrix<double, 2 * DimX + 1, DimX> sigma_points = x.transpose().replicate(2 * DimX + 1, 1); // initialize as a matrix containing state means
    
    mean_weights(0, 0) = lambda / (n + lambda);
    covariance_weights(0, 0) = mean_weights(0, 0) + 1 - alpha*alpha + beta;
    
    Eigen::Matrix<double, DimX, DimX> sqrt_matrix = ((n + lambda) * P).llt().matrixU();

    for (int i = 1; i <= DimX; i++) {
        sigma_points.row(i) += sqrt_matrix.row(i - 1);
    }

    for (int i = DimX + 1; i <= 2*DimX; i++) {
        sigma_points.row(i) -= sqrt_matrix.row(i - 1 - DimX);
    }

    for (int i = 1; i <= 2*DimX; i++) {
        mean_weights(0, i) = 1/(2*(n+lambda));
        covariance_weights(0, i) = mean_weights(0, i);
    }

    return {sigma_points, mean_weights, covariance_weights};
}

template <int DimX, int DimZ>
std::pair<Eigen::Matrix<double, DimX, 1>, Eigen::Matrix<double, DimX, DimX>>
UnscentedKalmanFilter<DimX, DimZ>::unscented_transform(
    Eigen::Matrix<double, 1, 2 * DimX + 1> mean_weights,
    Eigen::Matrix<double, 1, 2 * DimX + 1> covariance_weights,
    Eigen::Matrix<double, 2 * DimX + 1, DimX> sigma_points,
    Eigen::Matrix<double, DimX, DimX> noise)
{
    Eigen::Matrix<double, DimX, 1> ret_x = (mean_weights * sigma_points).transpose();
    Eigen::Matrix<double, DimX, DimX> ret_P = Eigen::Matrix<double, DimX, DimX>::Zero();
    for (int i = 0; i <= 2 * DimX; i++) {
        Eigen::Matrix<double, 1, DimX> distance = (Eigen::Matrix<double, 1, DimX>)(sigma_points.row(i)) - ret_x.transpose();
        ret_P += covariance_weights(0, i) * (distance.transpose() * distance);
    }
    ret_P += noise;
    return {ret_x, ret_P};
}
