#include "software/sensor_fusion/filter/ball_filter.h"

#include <Eigen/Dense>
#include <algorithm>
#include <vector>

#include "shared/constants.h"
#include "software/logger/logger.h"
#include "software/geom/algorithms/closest_point.h"
#include "software/geom/algorithms/contains.h"
#include "software/math/math_functions.h"

std::optional<Ball> BallFilter::estimateBallState(
    const std::vector<BallDetection>& new_ball_detections, const Rectangle& filter_area)
{
    addNewDetectionsToBuffer(new_ball_detections, filter_area);
    return Ball(Point(filter.x(0, 0), filter.x(2, 0)), Vector(filter.x(1, 0), filter.x(3, 0)), last_time.value());
}

void BallFilter::addNewDetectionsToBuffer(std::vector<BallDetection> new_ball_detections,
                                          const Rectangle& filter_area)
{
    // Sort the detections in increasing order before processing. This places the oldest
    // detections (with the smallest timestamp) at the front of the buffer, and the most
    // recent detections (largest timestamp) at the end of the buffer.
    std::sort(new_ball_detections.begin(), new_ball_detections.end());

    for (const auto& detection : new_ball_detections)
    {

        if (last_time.has_value()) {
            Duration time_diff =
                detection.timestamp - last_time.value();

            // Ignore any data from the past, and any data that is as old as the oldest
            // data in the buffer since it provides no additional value. This also
            // prevents division by 0 when calculating the estimated velocity
            if (time_diff.toSeconds() <= 0)
            {
                continue;
            }
            double delta_time_seconds = time_diff.toSeconds();
            double time_squared = delta_time_seconds * delta_time_seconds;
            filter.Q << time_squared / 2,   delta_time_seconds, 0,                  0,
                         delta_time_seconds, 1,                  0,                  0,
                         0,                  0,                  time_squared / 2,   delta_time_seconds,
                         0,                  0,                  delta_time_seconds, 1;
            filter.Q *= 0.4;
            filter.predict(time_diff.toSeconds());
            if (detection.position.has_value() && contains(filter_area, detection.position.value())) {
                Eigen::Matrix<double, 2, 1> z = Eigen::Matrix<double, 2, 1>::Zero();
                z << detection.position->x(), detection.position->y();

                filter.update(z);
            }
            last_time = std::optional(detection.timestamp);
        } else if (detection.position.has_value()){
            // this is the first detection, we can just hard set the position of the ball with low confidence
            filter.x << detection.position->x(), 0, detection.position->y(), 0;
            filter.P << 0.05, 0.0, 0.0, 0.0,
                        0.0, 0.05, 0.0, 0.0,
                        0.0, 0.0, 0.05, 0.0,
                        0.0, 0.0, 0.0, 0.05;
            last_time = std::optional(detection.timestamp);
        }

        
    }
}
