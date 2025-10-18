#pragma once

#include <boost/circular_buffer.hpp>
#include <optional>
#include <Eigen/Dense>
#include <functional>

#include "software/geom/line.h"
#include "software/geom/point.h"
#include "software/geom/rectangle.h"
#include "software/sensor_fusion/filter/vision_detection.h"
#include "software/sensor_fusion/filter/unscented_kalman_filter.hpp"
#include "software/time/timestamp.h"
#include "software/world/ball.h"

/**
 * Given ball data from SSL Vision, filters and returns the position/velocity of the
 * "real" ball.
 *
 * This ball filter stores a buffer of previous SSL Vision detections, and uses linear
 * regression to find the path the ball is travelling on and estimate its position
 * and velocity. This buffer/regression system was chosen because it results in a
 * very stable output, particularly for the ball velocity. The data we receive isn't
 * perfect (which is why we have a filter). If we receive a noisy position that is off
 * the ball's current trajectory, it will have minimal impact. This means that as
 * the ball is travelling, this filter will return a very steady velocity vector.
 * This is important because small deviations in velocity orientation can have large
 * effects when the AI tries to predict the future position of the ball. For example,
 * consistently receiving a pass relies on the ball's velocity being very stable,
 * otherwise the robot would "jiggle" back and forth as the estimated receiver position
 * would keep changing.
 */
class BallFilter
{
   public:
    // The min and max sizes of the ball detection buffer.
    // As the ball slows down, the buffer size will approach the MAX_BUFFER_SIZE.
    // As the ball speeds up, the buffer size will approach the MIN_BUFFER_SIZE.
    static constexpr unsigned int MIN_BUFFER_SIZE = 4;
    static constexpr unsigned int MAX_BUFFER_SIZE = 10;
    // If the estimated ball speed is less than this value, the largest possible buffer
    // will be used by the filter
    static constexpr double MIN_BUFFER_SIZE_VELOCITY_MAGNITUDE = 0.5;
    // If the estimated ball speed is greater than this value, the smallest possible
    // buffer will be used by the filter
    static constexpr double MAX_BUFFER_SIZE_VELOCITY_MAGNITUDE = 4.0;
    // The extra amount beyond the ball's max speed that we treat ball detections as valid
    static constexpr double MAX_ACCEPTABLE_BALL_SPEED_BUFFER = 2.0;
    // The maximum error threshold to considering using the generated linear regression
    // TODO (#2752): Investigate different values of error threshold
    static constexpr double LINEAR_REGRESSION_ERROR_THRESHOLD = 1000.0;

    /**
     * Creates a new Ball Filter
     */
    explicit BallFilter():
    filter(UnscentedKalmanFilter<4, 2>(Eigen::Matrix<double, 4, 1>::Zero(), Eigen::Matrix<double, 4, 4>::Zero(),
            Eigen::Matrix<double, 4, 4>::Zero(), Eigen::Matrix<double, 2, 4>::Zero(), Eigen::Matrix<double, 2, 2>::Zero(),
            [] (Eigen::Matrix<double, 1, 4> i, double delta) {
            i(0, 0) = i(0, 0) + i(0, 1) * delta;
            i(0, 2) = i(0, 2) + i(0, 3) * delta;
            return i;
        }, 0.1, 2.0, 3.0 - 4.0)) {
            filter.H << 1.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0;
            filter.Q << 1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0;
            filter.R << 0.05, 0.0,
                        0.0, 0.05;
    }

    /**
     * Update the filter with the new ball detection data, and returns the new
     * estimated state of the ball given the new data
     *
     * @param new_ball_detections A list of new Ball detections
     * @param filter_area The area within which the ball filter will work. Any detections
     * outside of this area will be ignored.
     *
     * @return The new ball based on the estimated state of the ball given the new data.
     * If a filtered result cannot be calculated, returns std::nullopt
     */
    std::optional<Ball> estimateBallState(
        const std::vector<BallDetection>& new_ball_detections,
        const Rectangle& filter_area);

   private:

   void addNewDetectionsToBuffer(std::vector<BallDetection> new_ball_detections,
                                          const Rectangle &filter_area);

    UnscentedKalmanFilter<4, 2> filter;

    std::optional<Timestamp> last_time = std::nullopt;
};
