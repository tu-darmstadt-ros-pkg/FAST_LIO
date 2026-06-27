#pragma once
// SLERP/LERP pose-based point cloud undistortion.
//
// Buffers stamped SE(3) poses from the SLAM system and undistorts each point
// to a common reference time using quaternion SLERP for rotation and LERP for
// translation.  No IMU integration is performed — used for the optional HQ
// secondary output where one extra scan of latency is acceptable.
//
// Ported from pointcloud_undistorter/internal/pose_undistorter.hpp.

#include <common_lib.h>

#include <Eigen/Geometry>
#include <deque>
#include <cstddef>
#include <mutex>

class PoseUndistorter {
public:
    // Add a timestamped pose from the SLAM odometry stream.
    // Out-of-order poses are silently dropped.
    void addPose(double stamp, const Eigen::Quaterniond& q, const Eigen::Vector3d& pos);

    // True when the buffer contains at least one pose before t_beg and one
    // at-or-after t_end, so every point in the scan can be interpolated.
    bool hasCoverage(double t_beg, double t_end) const;

    // Undistort cloud to reference time t_ref using SLERP + LERP.
    //   t_beg   : scan start time in seconds (to convert curvature ms offsets)
    //   t_ref   : target time (typically scan end)
    //   R_LI    : LiDAR-to-IMU rotation
    //   t_LI    : LiDAR-to-IMU translation
    // Points whose time falls outside the buffered range are left unchanged.
    // Curvature is zeroed on every successfully corrected point.
    // Returns false and leaves cloud unchanged if pose coverage is missing.
    bool undistort(PointCloudXYZI& cloud, double t_beg, double t_ref,
                   const M3D& R_LI, const V3D& t_LI);

    // Remove poses older than t to keep memory bounded.
    // Always keeps one pose before t so interpolation works near the boundary.
    void pruneBefore(double t);

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.clear();
    }

    // Timestamp of the most recent buffered pose, or -1.0 if empty.
    double newestStamp() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.empty() ? -1.0 : buffer_.back().stamp;
    }

private:
    struct StampedPose {
        double stamp;
        Eigen::Quaterniond q;
        Eigen::Vector3d pos;
    };

    std::deque<StampedPose> buffer_;
    static constexpr std::size_t kMaxPoses = 500;

    bool interpolate(double t, Eigen::Quaterniond& q_out, Eigen::Vector3d& pos_out) const;

    mutable std::mutex mutex_;
};

// ---- Inline implementations ------------------------------------------------

inline void PoseUndistorter::addPose(double stamp, const Eigen::Quaterniond& q,
                                     const Eigen::Vector3d& pos) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!buffer_.empty() && stamp <= buffer_.back().stamp) return;
    if (buffer_.size() >= kMaxPoses) buffer_.pop_front();
    buffer_.push_back({stamp, q.normalized(), pos});
}

inline bool PoseUndistorter::hasCoverage(double t_beg, double t_end) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_.size() < 2) return false;
    return buffer_.front().stamp <= t_beg && buffer_.back().stamp >= t_end;
}

inline void PoseUndistorter::pruneBefore(double t) {
    std::lock_guard<std::mutex> lock(mutex_);
    while (buffer_.size() > 1 && buffer_[1].stamp < t) buffer_.pop_front();
}

inline bool PoseUndistorter::interpolate(double t, Eigen::Quaterniond& q_out,
                                         Eigen::Vector3d& pos_out) const {
    if (buffer_.size() < 2 || t < buffer_.front().stamp || t > buffer_.back().stamp)
        return false;

    std::size_t lo = 0, hi = buffer_.size() - 1;
    while (hi - lo > 1) {
        const std::size_t mid = (lo + hi) / 2;
        if (buffer_[mid].stamp <= t) lo = mid; else hi = mid;
    }

    const double dt = buffer_[hi].stamp - buffer_[lo].stamp;
    const double alpha = (dt > 1e-9) ? (t - buffer_[lo].stamp) / dt : 0.0;

    q_out = buffer_[lo].q.slerp(alpha, buffer_[hi].q);
    pos_out = (1.0 - alpha) * buffer_[lo].pos + alpha * buffer_[hi].pos;
    return true;
}

inline bool PoseUndistorter::undistort(PointCloudXYZI& cloud, double t_beg, double t_ref,
                                        const M3D& R_LI, const V3D& t_LI) {
    std::lock_guard<std::mutex> lock(mutex_);
    Eigen::Quaterniond q_ref;
    Eigen::Vector3d pos_ref;
    if (!interpolate(t_ref, q_ref, pos_ref)) return false;

    const Eigen::Matrix3d R_ref = q_ref.toRotationMatrix();
    const Eigen::Matrix3d R_LI_T = R_LI.transpose();

    for (auto& pt : cloud.points) {
        const double t_i = t_beg + static_cast<double>(pt.curvature) / 1000.0;

        Eigen::Quaterniond q_i;
        Eigen::Vector3d pos_i;
        if (!interpolate(t_i, q_i, pos_i)) continue;

        // Relative SE(3) motion compensation: T_world(t_ref)^-1 * T_world(t_i)
        // 1. P → IMU frame at t_i: R_LI * P + t_LI
        // 2. → world frame:        R_i * P_imu_i + pos_i
        // 3. → IMU frame at t_ref: R_ref^T * (P_world - pos_ref)
        // 4. → LiDAR frame:        R_LI^T * (P_imu_ref - t_LI)
        const Eigen::Vector3d P_imu_i  = R_LI * Eigen::Vector3d(pt.x, pt.y, pt.z) + t_LI;
        const Eigen::Vector3d P_imu_ref =
            R_ref.transpose() * (q_i.toRotationMatrix() * P_imu_i + (pos_i - pos_ref));
        const Eigen::Vector3d P_lidar  = R_LI_T * (P_imu_ref - t_LI);

        pt.x = static_cast<float>(P_lidar.x());
        pt.y = static_cast<float>(P_lidar.y());
        pt.z = static_cast<float>(P_lidar.z());
        pt.curvature = 0.0f;
    }
    return true;
}
