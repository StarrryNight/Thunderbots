#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <functional>
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
     * Creates kalman filter with all matrices and vectors set to zero.
     */
    UnscentedKalmanFilter(int i, int k)
    {
        x = Eigen::Matrix<double, DimX, 1>();
    }

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
    std::function<Eigen::Matrix<double, DimX, 1>(Eigen::Matrix<double, DimX, 1>, double)> process_function  // Process model
    std::function<Eigen::Matrix<double, DimZ, 1>(Eigen::Matrix<double, DimX, 1>)> measurement_function  // Converts state to measurement space

private:
    struct SigmaPoint {
        Eigen::Matrix<double, DimX, 1> position;
        int weight;
    };

    // PARAMETERS FOR THE VAN DER MERWE'S ALGORITHM FOR SIGMA POINT GENERATION
    // variable names are the same as in the literature
    // usually n is set to 2, 0 < alpha <= 1, beta = 2, kappa = 3 - DimX
    // hence total sigma point count will be 1 + n * DimX
    int n; // sigma points to generate per axis (+ mean)
    int alpha;
    int beta;
    int kappa;


    /**
     * Uses Van der Merwe's Scaled Sigma Point Algorithm to generate Sigma points to pass through the process function
     *
     *
     */
    std::vector<SigmaPoint> getSigmaPoints();
};

template <int dimX, int dimZ>
void predict(double input) {
    std::vector<Eigen::Matrix<double, DimX, 1>> transformed_points = {};
    for (auto sigma_point : getSigmaPoints()) {
        transformed_points.push_back(process_function(sigma_point.position, input));
    }
}

template <int DimX, int DimZ>
void KalmanFilter<DimX, DimZ, DimU>::update(Eigen::Matrix<double, DimZ, 1> z)
{

}
