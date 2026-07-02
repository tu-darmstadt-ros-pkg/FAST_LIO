#ifndef L2_IMU_UNDISTORT_H
#define L2_IMU_UNDISTORT_H

#include <deque>
#include <vector>
#include <common_lib.h>
#include <sensor_msgs/msg/imu.hpp>
#include "use-ikfom.hpp"

/*** Undistortion of the secondary lidar (L2) with its own IMU (I2).
 *
 * The rotation over the scan is integrated from I2's own gyro, so errors in the
 * L2-to-L1 extrinsic can no longer smear the compensation rotation. Only the
 * (small) intra-scan translation is derived from the primary IMU trajectory via
 * the lever arm, where extrinsic errors enter second-order. ***/

// Extrinsics needed to undistort L2 with I2 (p_I2 = R_I2_L2 * p_L2 + t_I2_L2).
struct L2ImuExtrinsics
{
  M3D R_I2_L2 = Eye3d;
  V3D t_I2_L2 = Zero3d;
  M3D lidar2_R_wrt_L1 = Eye3d;
  V3D lidar2_T_wrt_L1 = Zero3d;
};

// Value-copy of the L2 undistortion trajectory for one scan.
// All fields are owned copies — thread-safe to use from any thread.
// Knot semantics match ImuProcess::IMUpose: pos/vel/rot describe the knot itself
// (rot = gyro2-integrated rotation in an arbitrary common start frame), acc/gyr
// describe the segment ending at the knot. offset_time is relative to the L2
// scan begin, i.e. the same base as the cloud's per-point curvature [ms].
struct L2TrajectorySnapshot
{
  vector<Pose6D> knots;
  M3D R_ref_T;       // transpose of the gyro2-integrated rotation at the target time
  M3D R_W_I2_ref_T;  // transpose of the world rotation of I2 at the target time
  V3D p_I2_ref;      // world position of I2 at the target time
  M3D R_I2_L2;
  V3D t_I2_L2;

  bool empty() const { return knots.empty(); }

  // Compensate every point to the target time, in the native L2 frame.
  void applyTo(PointCloudXYZI &pcl_in_out) const
  {
    if (knots.empty() || pcl_in_out.points.empty()) return;
    sort(pcl_in_out.points.begin(), pcl_in_out.points.end(),
         [](const PointType &a, const PointType &b) { return a.curvature < b.curvature; });

    V3D angvel_avr, acc_imu, vel_imu, pos_imu;
    M3D R_rel;
    double dt = 0;

    auto it_pcl = pcl_in_out.points.end() - 1;
    for (auto it_kp = knots.end() - 1; it_kp != knots.begin(); it_kp--)
    {
      auto head = it_kp - 1;
      auto tail = it_kp;
      R_rel << MAT_FROM_ARRAY(head->rot);
      vel_imu << VEC_FROM_ARRAY(head->vel);
      pos_imu << VEC_FROM_ARRAY(head->pos);
      acc_imu << VEC_FROM_ARRAY(tail->acc);
      angvel_avr << VEC_FROM_ARRAY(tail->gyr);

      for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--)
      {
        dt = it_pcl->curvature / double(1000) - head->offset_time;
        M3D R_i(R_rel * Exp(angvel_avr, dt));
        V3D P_pt(it_pcl->x, it_pcl->y, it_pcl->z);
        V3D p_i(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt);
        V3D P_compensate = R_I2_L2.transpose() *
            (R_ref_T * (R_i * (R_I2_L2 * P_pt + t_I2_L2)) + R_W_I2_ref_T * (p_i - p_I2_ref) - t_I2_L2);

        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);

        if (it_pcl == pcl_in_out.points.begin()) break;
      }
    }
  }
};

class L2ImuUndistort
{
 public:
  void setExtrinsics(const L2ImuExtrinsics &ext) { ext_ = ext; }
  const L2ImuExtrinsics &extrinsics() const { return ext_; }

  void resetInit()
  {
    gyro_bias_ = Zero3d;
    init_count_ = 0;
    bias_checked_ = false;
  }

  // Accumulate the gyro bias while the platform is stationary during filter init.
  void feedInit(const deque<sensor_msgs::msg::Imu::ConstSharedPtr> &imu2)
  {
    for (const auto &msg : imu2)
    {
      V3D gyr(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
      if (!gyr.allFinite()) continue;
      init_count_++;
      gyro_bias_ += (gyr - gyro_bias_) / init_count_;
    }
    if (!bias_checked_ && biasReady())
    {
      bias_checked_ = true;
      if (gyro_bias_.norm() > 0.1)
        printf("\033[1;31m[IMU2] Init gyro bias %.3f rad/s > 0.1 — robot moving during init? "
               "L2 undistortion will be degraded\n\033[0m", gyro_bias_.norm());
      else
        printf("[IMU2] Init done, gyro bias [%.4f, %.4f, %.4f] rad/s\n",
               gyro_bias_(0), gyro_bias_(1), gyro_bias_(2));
    }
  }

  bool biasReady() const { return init_count_ >= MIN_INIT_SAMPLES; }

  // Build the undistortion trajectory for the current L2 scan: rotation from the
  // gyro2 measurements in meas.imu2, translation resampled from the primary IMU
  // trajectory (imu1_poses, offsets relative to imu1_pose_base_time). end_state
  // must be the filter state at target_time. Returns false when the IMU2 data
  // does not cover [scan begin, target_time] — the caller falls back to the
  // legacy transform-first path.
  bool buildSnapshot(const MeasureGroup &meas, const vector<Pose6D> &imu1_poses,
                     double imu1_pose_base_time, const state_ikfom &end_state,
                     double target_time, L2TrajectorySnapshot &out) const
  {
    const auto &imu2 = meas.imu2;
    const double scan_beg_time = meas.lidar_is_l2 ? meas.lidar_beg_time : meas.lidar_beg_time2;
    if (!biasReady() || imu2.size() < 2 || imu1_poses.size() < 2) return false;
    if (get_time_sec(imu2.front()->header.stamp) > scan_beg_time + 0.02) return false;
    if (get_time_sec(imu2.back()->header.stamp) < target_time - 0.1) return false;

    /*** world rotation/position of I2 at the target time via the extrinsic chain ***/
    const M3D offset_R = end_state.offset_R_L_I.toRotationMatrix();  // L1 -> I1
    const M3D R_I1_I2 = offset_R * ext_.lidar2_R_wrt_L1 * ext_.R_I2_L2.transpose();
    const V3D t_I1_I2 = offset_R * (ext_.lidar2_T_wrt_L1 -
                        ext_.lidar2_R_wrt_L1 * (ext_.R_I2_L2.transpose() * ext_.t_I2_L2)) +
                        end_state.offset_T_L_I;
    const M3D R_W_I1_end = end_state.rot.toRotationMatrix();
    const M3D R_W_I2_ref = R_W_I1_end * R_I1_I2;

    out.R_I2_L2 = ext_.R_I2_L2;
    out.t_I2_L2 = ext_.t_I2_L2;
    out.R_W_I2_ref_T = R_W_I2_ref.transpose();
    out.p_I2_ref = end_state.pos + R_W_I1_end * t_I1_I2;

    /*** translation of I2 resampled from the primary trajectory via the lever arm;
     * second-order lever-arm terms (angular acceleration, centripetal) are ignored ***/
    size_t seg = 0;  // primary segment cursor, advanced monotonically
    auto sample_primary = [&](double t, V3D &pos, V3D &vel, V3D &acc) {
      double toff = t - imu1_pose_base_time;
      while (seg + 2 < imu1_poses.size() && imu1_poses[seg + 1].offset_time <= toff) seg++;
      const Pose6D &head = imu1_poses[seg];
      const Pose6D &tail = imu1_poses[seg + 1];
      double dt = toff - head.offset_time;
      M3D R_head; R_head << MAT_FROM_ARRAY(head.rot);
      V3D vel_h; vel_h << VEC_FROM_ARRAY(head.vel);
      V3D pos_h; pos_h << VEC_FROM_ARRAY(head.pos);
      V3D acc_t; acc_t << VEC_FROM_ARRAY(tail.acc);
      V3D gyr_t; gyr_t << VEC_FROM_ARRAY(tail.gyr);
      M3D R_I1 = R_head * Exp(gyr_t, dt);
      pos = pos_h + vel_h * dt + 0.5 * acc_t * dt * dt + R_I1 * t_I1_I2;
      vel = vel_h + acc_t * dt + R_I1 * gyr_t.cross(t_I1_I2);
      acc = acc_t;
    };

    /*** gyro2 relative rotation trajectory, forward from identity at the first message ***/
    out.knots.clear();
    out.knots.reserve(imu2.size() + 1);
    M3D R_rel = Eye3d;
    double t_prev = get_time_sec(imu2.front()->header.stamp);
    V3D pos_i2, vel_i2, acc_i2;
    sample_primary(t_prev, pos_i2, vel_i2, acc_i2);
    out.knots.push_back(set_pose6d(t_prev - scan_beg_time, acc_i2, Zero3d, vel_i2, pos_i2, Eye3d));

    V3D angvel_avr = Zero3d;
    for (size_t k = 1; k < imu2.size(); k++)
    {
      const auto &lo = imu2[k - 1];
      const auto &hi = imu2[k];
      double t_k = get_time_sec(hi->header.stamp);
      double dt = t_k - t_prev;
      angvel_avr << 0.5 * (lo->angular_velocity.x + hi->angular_velocity.x),
                    0.5 * (lo->angular_velocity.y + hi->angular_velocity.y),
                    0.5 * (lo->angular_velocity.z + hi->angular_velocity.z);
      if (!angvel_avr.allFinite() || dt <= 0.0) continue;
      if (dt > 0.1) return false;  // inner gap in the IMU2 stream
      angvel_avr -= gyro_bias_;
      R_rel = R_rel * Exp(angvel_avr, dt);
      sample_primary(t_k, pos_i2, vel_i2, acc_i2);
      out.knots.push_back(set_pose6d(t_k - scan_beg_time, acc_i2, angvel_avr, vel_i2, pos_i2, R_rel));
      t_prev = t_k;
    }
    if (out.knots.size() < 2) return false;

    /*** extend to the target time with the last message's rate ***/
    const auto &last = imu2.back();
    V3D gyr_last(last->angular_velocity.x, last->angular_velocity.y, last->angular_velocity.z);
    gyr_last -= gyro_bias_;
    double dt_f = target_time - t_prev;
    M3D R_ref = R_rel;
    if (dt_f > 0.0)
    {
      R_ref = R_rel * Exp(gyr_last, dt_f);
      sample_primary(target_time, pos_i2, vel_i2, acc_i2);
      out.knots.push_back(set_pose6d(target_time - scan_beg_time, acc_i2, gyr_last, vel_i2, pos_i2, R_ref));
    }
    out.R_ref_T = R_ref.transpose();
    return true;
  }

 private:
  static constexpr int MIN_INIT_SAMPLES = 100;  // ~0.5 s at 200 Hz, robot stationary at startup
  L2ImuExtrinsics ext_;
  V3D gyro_bias_ = Zero3d;
  int init_count_ = 0;
  bool bias_checked_ = false;
};

#endif  // L2_IMU_UNDISTORT_H
