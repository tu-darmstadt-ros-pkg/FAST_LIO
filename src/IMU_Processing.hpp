#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <memory>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include "use-ikfom.hpp"
#include "l2_imu_undistort.hpp"

/// *************Preconfiguration

#define MAX_INI_COUNT (10)

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

// Value-copy of the IMU trajectory built during a Process() call.
// All fields are owned copies — thread-safe to use from any thread concurrently
// with the next Process() call.
struct TrajectorySnapshot {
    vector<Pose6D> poses;
    state_ikfom final_state;
    bool empty() const { return poses.empty(); }
    void applyTo(PointCloudXYZI& pcl_in_out) const;
};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  // void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void Reset(double start_timestamp, const sensor_msgs::msg::Imu::ConstSharedPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  void swap_lidar_end_time() { std::swap(last_lidar_end_time_, last_lidar_end_time_L2_); }
  double get_lidar_end_time_L2() const { return last_lidar_end_time_L2_; }
  void reset_lidar_end_time_L2(double t) { last_lidar_end_time_L2_ = t; }
  Eigen::Matrix<double, 12, 12> Q;
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, const PointCloudXYZI::Ptr& pcl_un_, const PointCloudXYZI::Ptr& pcl_L1_out, const PointCloudXYZI::Ptr& pcl_L2_out, const bool &multi_lidar = false);

  bool is_initialized() const { return !imu_need_init_; }

  // SLAM odometry anchor: resets EKF pos/rot/vel at scan start, bounding drift to one window.
  void setAnchorPose(const V3D& pos, const M3D& rot, const V3D& vel, double stamp);
  void clearAnchor();
  double anchor_stamp() const { std::lock_guard<std::mutex> l(anchor_mutex_); return anchor_stamp_; }

  // Value-copy of the IMU trajectory from the last Process() call — safe to use from any thread.
  TrajectorySnapshot captureTrajectory() const { return {IMUpose, last_undistort_state_}; }

  // Secondary-IMU undistortion: L2 scans are undistorted in their own frame with
  // their own IMU's gyro. Active once enableSecondaryImu() has been called.
  void enableSecondaryImu(const L2ImuExtrinsics &ext)
  {
    if (!l2_undistort_) l2_undistort_ = std::make_unique<L2ImuUndistort>();
    l2_undistort_->setExtrinsics(ext);
  }
  bool secondaryImuEnabled() const { return l2_undistort_ != nullptr; }
  // True when the last Process() undistorted L2 with its own IMU (the per-lidar L2
  // output is then in the native L2 frame).
  bool hasL2Snapshot() const { return l2_snapshot_valid_; }
  L2TrajectorySnapshot captureL2Trajectory() const { return l2_snapshot_; }
  ofstream fout_imu;
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double first_lidar_time;

 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  void RewindImuPose(const deque<sensor_msgs::msg::Imu::ConstSharedPtr> &v_imu, const state_ikfom &state, double pcl_beg_time);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out, const PointCloudXYZI::Ptr &pcl_native_l2_out = nullptr);
  void UndistortPclMultiLiDAR(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out, PointCloudXYZI::Ptr pcl_L1_out, PointCloudXYZI::Ptr pcl_L2_out);

  PointCloudXYZI::Ptr cur_pcl_un_;
  // sensor_msgs::ImuConstPtr last_imu_;
  sensor_msgs::msg::Imu::ConstSharedPtr last_imu_;
  deque<sensor_msgs::msg::Imu::ConstSharedPtr> v_imu_;
  vector<Pose6D> IMUpose;
  vector<M3D>    v_rot_pcl_;
  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  double start_timestamp_;
  double last_lidar_end_time_;
  double last_lidar_end_time_L2_ = 0.0;
  // Absolute time the filter state was last predicted to. Unlike the per-lidar
  // last_lidar_end_time_ cursors, this is global across both lidars.
  double state_time_ = 0.0;
  int    init_iter_num = 1;
  bool   b_first_frame_ = true;
  bool   imu_need_init_ = true;

  state_ikfom last_undistort_state_{};

  // Secondary-IMU undistortion (null when disabled).
  std::unique_ptr<L2ImuUndistort> l2_undistort_;
  L2TrajectorySnapshot l2_snapshot_;
  bool l2_snapshot_valid_ = false;
  int l2_fallback_count_ = 0;

  // Anchor pose from external SLAM odometry (protected by anchor_mutex_).
  mutable std::mutex anchor_mutex_;
  bool anchor_valid_{false};
  V3D anchor_pos_{Zero3d};
  M3D anchor_rot_{Eye3d};
  V3D anchor_vel_{Zero3d};
  double anchor_stamp_{-1.0};
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1),
      last_lidar_end_time_(0.0)
{
  init_iter_num = 1;
  Q = process_noise_cov();
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last     = Zero3d;
  acc_s_last      = Zero3d;
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;
  last_imu_.reset(new sensor_msgs::msg::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset()
{
  printf("\033[1;33m\n");
  printf("############################################################\n");
  printf("  IMU PROCESSOR RESET\n");
  printf("############################################################\n");
  printf("\033[0m\n");
  mean_acc          = V3D(0, 0, -1.0);
  mean_gyr          = V3D(0, 0, 0);
  angvel_last          = Zero3d;
  acc_s_last           = Zero3d;
  imu_need_init_       = true;
  b_first_frame_       = true;
  start_timestamp_     = -1;
  init_iter_num        = 1;
  last_lidar_end_time_    = 0.0;
  last_lidar_end_time_L2_ = 0.0;
  state_time_             = 0.0;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::msg::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
  l2_snapshot_valid_ = false;
  if (l2_undistort_) l2_undistort_->resetInit();
}

void ImuProcess::setAnchorPose(const V3D& pos, const M3D& rot, const V3D& vel, double stamp) {
    std::lock_guard<std::mutex> lock(anchor_mutex_);
    anchor_pos_ = pos;
    anchor_rot_ = rot;
    anchor_vel_ = vel;
    anchor_stamp_ = stamp;
    anchor_valid_ = true;
}

void ImuProcess::clearAnchor() {
    std::lock_guard<std::mutex> lock(anchor_mutex_);
    anchor_valid_ = false;
}

void TrajectorySnapshot::applyTo(PointCloudXYZI& pcl_in_out) const {
    if (poses.empty() || pcl_in_out.points.empty()) return;
    sort(pcl_in_out.points.begin(), pcl_in_out.points.end(), time_list);

    const state_ikfom& imu_state = final_state;
    V3D angvel_avr, acc_imu, vel_imu, pos_imu;
    M3D R_imu;
    double dt = 0;

    auto it_pcl = pcl_in_out.points.end() - 1;
    for (auto it_kp = poses.end() - 1; it_kp != poses.begin(); it_kp--) {
        auto head = it_kp - 1;
        auto tail = it_kp;
        R_imu << MAT_FROM_ARRAY(head->rot);
        vel_imu << VEC_FROM_ARRAY(head->vel);
        pos_imu << VEC_FROM_ARRAY(head->pos);
        acc_imu << VEC_FROM_ARRAY(tail->acc);
        angvel_avr << VEC_FROM_ARRAY(tail->gyr);

        for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
            dt = it_pcl->curvature / double(1000) - head->offset_time;
            M3D R_i(R_imu * Exp(angvel_avr, dt));
            V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
            V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
            V3D P_compensate =
                imu_state.offset_R_L_I.conjugate() *
                (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) -
                 imu_state.offset_T_L_I);
            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);
            if (it_pcl == pcl_in_out.points.begin()) break;
        }
    }
}

void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    first_lidar_time = meas.lidar_beg_time;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N ++;
  }
  double acc_norm = mean_acc.norm();
  if (fabs(acc_norm - G_m_s2) > 0.5)
  {
    printf("\033[1;31m\n");
    printf("!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=\n");
    printf("  IMU INIT WARNING: acc_norm=%.3f m/s^2  expected g=%.3f\n", acc_norm, G_m_s2);
    printf("  Robot may be moving during init — gravity direction UNRELIABLE\n");
    printf("!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=\n");
    printf("\033[0m\n");
  }
  else
  {
    printf("\033[1;32m\n");
    printf("============================================================\n");
    printf("  IMU INIT OK: acc_norm=%.3f m/s^2  (g=%.3f)  grav dir good\n", acc_norm, G_m_s2);
    printf("============================================================\n");
    printf("\033[0m\n");
  }

  state_ikfom init_state = kf_state.get_x();
  init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);
  
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr;
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state.change_x(init_state);

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity();
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  init_P(21,21) = init_P(22,22) = 0.00001;
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();

}

/*** Reconstruct IMU poses covering [pcl_beg_time, state_time_] by integrating the
 * measurements backward from the current filter state. In multi-lidar operation the
 * scan windows of the two lidars overlap, so the filter has already been propagated
 * past this scan's begin time — the undistortion trajectory must still cover the
 * full scan window. Appends knots to IMUpose in ascending time order, ending with
 * one at state_time_. Never touches the filter state or covariance. ***/
void ImuProcess::RewindImuPose(const deque<sensor_msgs::msg::Imu::ConstSharedPtr> &v_imu, const state_ikfom &state, double pcl_beg_time)
{
  int last_below = -1;
  for (size_t i = 0; i < v_imu.size(); i++)
  {
    if (rclcpp::Time(v_imu[i]->header.stamp).seconds() >= state_time_) break;
    last_below = static_cast<int>(i);
  }

  V3D grav(state.grav[0], state.grav[1], state.grav[2]);
  double t_hi = state_time_;
  M3D R_hi = state.rot.toRotationMatrix();
  V3D vel_hi = state.vel;
  V3D pos_hi = state.pos;
  vector<Pose6D> knots;  // built newest-first, appended in reverse below
  for (int j = last_below; j >= 0; j--)
  {
    const auto &lo = v_imu[j];
    const auto &hi = v_imu[(j + 1 < static_cast<int>(v_imu.size())) ? j + 1 : j];
    double t_lo = rclcpp::Time(lo->header.stamp).seconds();
    double dt = t_hi - t_lo;
    V3D angvel_avr(0.5 * (lo->angular_velocity.x + hi->angular_velocity.x),
                   0.5 * (lo->angular_velocity.y + hi->angular_velocity.y),
                   0.5 * (lo->angular_velocity.z + hi->angular_velocity.z));
    V3D acc_avr(0.5 * (lo->linear_acceleration.x + hi->linear_acceleration.x),
                0.5 * (lo->linear_acceleration.y + hi->linear_acceleration.y),
                0.5 * (lo->linear_acceleration.z + hi->linear_acceleration.z));
    if (!acc_avr.allFinite() || !angvel_avr.allFinite() || dt <= 0.0) continue;
    angvel_avr -= state.bg;
    V3D acc_w = R_hi * (acc_avr * G_m_s2 / mean_acc.norm() - state.ba) + grav;
    // the knot at t_hi is the tail of segment [t_lo, t_hi]: it carries the segment's acc/gyr
    knots.push_back(set_pose6d(t_hi - pcl_beg_time, acc_w, angvel_avr, vel_hi, pos_hi, R_hi));
    M3D R_lo = R_hi * Exp(angvel_avr, -dt);
    V3D vel_lo = vel_hi - acc_w * dt;
    V3D pos_lo = pos_hi - vel_lo * dt - 0.5 * acc_w * dt * dt;
    t_hi = t_lo; R_hi = R_lo; vel_hi = vel_lo; pos_hi = pos_lo;
  }
  // bottom knot; never used as a segment tail, so acc/gyr are placeholders
  knots.push_back(set_pose6d(t_hi - pcl_beg_time, acc_s_last, angvel_last, vel_hi, pos_hi, R_hi));
  IMUpose.insert(IMUpose.end(), knots.rbegin(), knots.rend());
}

void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out, const PointCloudXYZI::Ptr &pcl_native_l2_out)
{
  // Reset EKF pos/rot/vel from SLAM anchor if available.
  {
    std::lock_guard<std::mutex> lock(anchor_mutex_);
    if (anchor_valid_) {
      state_ikfom s = kf_state.get_x();
      s.pos = anchor_pos_;
      s.rot = anchor_rot_;
      s.vel = anchor_vel_;
      kf_state.change_x(s);
    }
  }

  /*** the imu covering the scan window (sync keeps one message at or before the scan begin) ***/
  auto v_imu = meas.imu;
  if (v_imu.empty()) v_imu.push_back(last_imu_);
  const double &imu_end_time = rclcpp::Time(v_imu.back()->header.stamp).seconds();
  const double &pcl_beg_time = meas.lidar_beg_time;
  const double &pcl_end_time = meas.lidar_end_time;

  /*** skip gap IMU to avoid dead-reckoning over a sensor restart ***/
  if (state_time_ > 0.0 && pcl_beg_time - state_time_ > 1.0)
  {
    printf("\033[1;33m[IMU] Gap %.3fs > 1s: skipping gap IMU, advancing cursor to scan start\n\033[0m",
           pcl_beg_time - state_time_);
    state_time_ = pcl_beg_time;
    last_lidar_end_time_ = pcl_beg_time;
  }

  /*** sort point clouds by offset time ***/
  pcl_in_out = *(meas.lidar);
  sort(pcl_in_out.points.begin(), pcl_in_out.points.end(), time_list);

  /*** Initialize IMU pose: rewind to cover [scan begin, filter time] when the filter
   * is already past this scan's begin (overlapping multi-lidar windows) ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.reserve(v_imu.size() + 2);
  if (state_time_ > 0.0 && pcl_beg_time < state_time_)
    RewindImuPose(v_imu, imu_state, pcl_beg_time);
  if (IMUpose.empty())
    IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;

  input_ikfom in;
  in.acc  << last_imu_->linear_acceleration.x, last_imu_->linear_acceleration.y, last_imu_->linear_acceleration.z;
  in.gyro << last_imu_->angular_velocity.x,    last_imu_->angular_velocity.y,    last_imu_->angular_velocity.z;
  in.acc = in.acc * G_m_s2 / mean_acc.norm();

  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    double tail_stamp = rclcpp::Time(tail->header.stamp).seconds();
    double head_stamp = rclcpp::Time(head->header.stamp).seconds();

    if (tail_stamp <= state_time_)    continue;

    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    if (!acc_avr.allFinite() || !angvel_avr.allFinite()) continue;

    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

    if(head_stamp < state_time_)
    {
      dt = tail_stamp - state_time_;
    }
    else
    {
      dt = tail_stamp - head_stamp;
    }

    in.acc = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    kf_state.predict(dt, Q, in);

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail_stamp - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  if (fabs(dt) > 0.2)
    printf("\033[1;33m[IMU] Large end-gap dt=%.3fs  pcl_end=%.3f  imu_end=%.3f  last_lidar_end=%.3f\n\033[0m",
           dt, pcl_end_time, imu_end_time, last_lidar_end_time_);
  kf_state.predict(dt, Q, in);

  imu_state = kf_state.get_x();
  if (!meas.imu.empty()) last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;
  state_time_ = std::max(state_time_, pcl_end_time);

  if (pcl_in_out.points.begin() == pcl_in_out.points.end()) return;

  /*** async L2 scan with its own IMU: undistort in the native L2 frame, then bring
   * the cloud into the L1 frame for the EKF update ***/
  if (meas.lidar_is_l2 && l2_undistort_)
  {
    l2_snapshot_valid_ = l2_undistort_->buildSnapshot(meas, IMUpose, pcl_beg_time, imu_state, pcl_end_time, l2_snapshot_);
    const auto &ext = l2_undistort_->extrinsics();
    if (l2_snapshot_valid_)
    {
      l2_snapshot_.applyTo(pcl_in_out);
      if (pcl_native_l2_out) *pcl_native_l2_out = pcl_in_out;
      for (auto &pt : pcl_in_out.points)
      {
        V3D p_L1 = ext.lidar2_R_wrt_L1 * V3D(pt.x, pt.y, pt.z) + ext.lidar2_T_wrt_L1;
        pt.x = p_L1(0); pt.y = p_L1(1); pt.z = p_L1(2);
      }
      last_undistort_state_ = imu_state;
      return;
    }
    /*** fallback: the legacy path below expects the cloud in the L1 frame ***/
    if (++l2_fallback_count_ % 100 == 1)
      printf("\033[1;33m[IMU2] insufficient IMU2 data — falling back to primary-IMU undistortion for L2 (%d so far)\n\033[0m",
             l2_fallback_count_);
    for (auto &pt : pcl_in_out.points)
    {
      V3D p_L1 = ext.lidar2_R_wrt_L1 * V3D(pt.x, pt.y, pt.z) + ext.lidar2_T_wrt_L1;
      pt.x = p_L1(0); pt.y = p_L1(1); pt.z = p_L1(2);
    }
  }

  /*** undistort each lidar point (backward propagation) ***/
  auto it_pcl = pcl_in_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    acc_imu<<VEC_FROM_ARRAY(tail->acc);
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
      V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!
      
      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_in_out.points.begin()) break;
    }
  }
  last_undistort_state_ = imu_state;
}

void ImuProcess::UndistortPclMultiLiDAR(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out, PointCloudXYZI::Ptr pcl_L1_out, PointCloudXYZI::Ptr pcl_L2_out)
{
  if (meas.lidar_beg_time2 <= 0.0 || meas.lidar2 == nullptr || meas.lidar2->empty()) {
    std::cerr << "[ImuProcess] UndistortPclMultiLiDAR: invalid L2 measurement, falling back to single-lidar\n";
    UndistortPcl(meas, kf_state, pcl_in_out);
    return;
  }

  // Reset EKF pos/rot/vel from SLAM anchor if available.
  {
    std::lock_guard<std::mutex> lock(anchor_mutex_);
    if (anchor_valid_) {
      state_ikfom s = kf_state.get_x();
      s.pos = anchor_pos_;
      s.rot = anchor_rot_;
      s.vel = anchor_vel_;
      kf_state.change_x(s);
    }
  }

  /*** the imu covering the combined scan window (sync keeps one message at or before its begin) ***/
  auto v_imu = meas.imu;
  if (v_imu.empty()) v_imu.push_back(last_imu_);
  const double &imu_end_time = rclcpp::Time(v_imu.back()->header.stamp).seconds();
  const double pcl_beg_time = std::min(meas.lidar_beg_time, meas.lidar_beg_time2);
  const double pcl_end_time = std::max(meas.lidar_end_time, meas.lidar_end_time2);

  /*** skip gap IMU to avoid dead-reckoning over a sensor restart ***/
  if (state_time_ > 0.0 && pcl_beg_time - state_time_ > 1.0)
  {
    printf("\033[1;33m[IMU/multi] Gap %.3fs > 1s: skipping gap IMU, advancing cursor to scan start\n\033[0m",
           pcl_beg_time - state_time_);
    state_time_ = pcl_beg_time;
    last_lidar_end_time_ = pcl_beg_time;
  }

  /*** sort point clouds by offset time (adjust curvature relative to combined beg time) ***/
  *pcl_L1_out = *(meas.lidar);
  *pcl_L2_out = *(meas.lidar2);

  double time_offset_1 = (meas.lidar_beg_time - pcl_beg_time) * 1000.0;
  for (auto& pt : pcl_L1_out->points) pt.curvature += time_offset_1;

  // With the secondary IMU, L2 is undistorted in its own frame and time base; only
  // the legacy path rebases to the combined base (fallback rebasing happens later).
  const bool try_l2_own_imu = l2_undistort_ != nullptr;
  double time_offset_2 = (meas.lidar_beg_time2 - pcl_beg_time) * 1000.0;
  if (!try_l2_own_imu)
    for (auto& pt : pcl_L2_out->points) pt.curvature += time_offset_2;

  sort(pcl_L1_out->points.begin(), pcl_L1_out->points.end(), time_list);
  sort(pcl_L2_out->points.begin(), pcl_L2_out->points.end(), time_list);

  /*** Initialize IMU pose: rewind to cover [combined begin, filter time] (the previous
   * combined window's end generally lies past this one's begin) ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.reserve(v_imu.size() + 2);
  if (state_time_ > 0.0 && pcl_beg_time < state_time_)
    RewindImuPose(v_imu, imu_state, pcl_beg_time);
  if (IMUpose.empty())
    IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;
  double dt = 0;
  input_ikfom in;
  in.acc  << last_imu_->linear_acceleration.x, last_imu_->linear_acceleration.y, last_imu_->linear_acceleration.z;
  in.gyro << last_imu_->angular_velocity.x,    last_imu_->angular_velocity.y,    last_imu_->angular_velocity.z;
  in.acc = in.acc * G_m_s2 / mean_acc.norm();

  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    double tail_stamp = rclcpp::Time(tail->header.stamp).seconds();
    double head_stamp = rclcpp::Time(head->header.stamp).seconds();

    if (tail_stamp <= state_time_)    continue;

    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    if (!acc_avr.allFinite() || !angvel_avr.allFinite()) continue;

    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm();

    if(head_stamp < state_time_)
      dt = tail_stamp - state_time_;
    else
      dt = tail_stamp - head_stamp;

    in.acc = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    kf_state.predict(dt, Q, in);

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail_stamp - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  if (fabs(dt) > 0.2)
    printf("\033[1;33m[IMU/multi] Large end-gap dt=%.3fs  pcl_end=%.3f  imu_end=%.3f  last_lidar_end=%.3f\n\033[0m",
           dt, pcl_end_time, imu_end_time, last_lidar_end_time_);
  kf_state.predict(dt, Q, in);

  imu_state = kf_state.get_x();
  if (!meas.imu.empty()) last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;
  state_time_ = std::max(state_time_, pcl_end_time);

  /*** L2 with its own IMU: build the gyro2 trajectory targeting the combined scan end;
   * on failure fall back to the legacy path (L1 frame + combined time base) ***/
  bool l2_own_imu = false;
  if (try_l2_own_imu)
  {
    l2_snapshot_valid_ = l2_undistort_->buildSnapshot(meas, IMUpose, pcl_beg_time, imu_state, pcl_end_time, l2_snapshot_);
    l2_own_imu = l2_snapshot_valid_;
    if (!l2_own_imu)
    {
      if (++l2_fallback_count_ % 100 == 1)
        printf("\033[1;33m[IMU2] insufficient IMU2 data — falling back to primary-IMU undistortion for L2 (%d so far)\n\033[0m",
               l2_fallback_count_);
      const auto &ext = l2_undistort_->extrinsics();
      for (auto& pt : pcl_L2_out->points)
      {
        V3D p_L1 = ext.lidar2_R_wrt_L1 * V3D(pt.x, pt.y, pt.z) + ext.lidar2_T_wrt_L1;
        pt.x = p_L1(0); pt.y = p_L1(1); pt.z = p_L1(2);
        pt.curvature += time_offset_2;
      }
    }
  }

  /*** undistort L1 and L2 points in parallel (independent backward propagations) ***/
  #pragma omp parallel sections num_threads(2)
  {
    #pragma omp section
    {
      if (!pcl_L1_out->points.empty())
      {
        V3D angvel_l1, acc_l1, vel_l1, pos_l1;
        M3D R_l1;
        double dt_l1 = 0;
        auto it_pcl = pcl_L1_out->points.end() - 1;
        for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
        {
          auto head = it_kp - 1;
          auto tail = it_kp;
          R_l1<<MAT_FROM_ARRAY(head->rot);
          vel_l1<<VEC_FROM_ARRAY(head->vel);
          pos_l1<<VEC_FROM_ARRAY(head->pos);
          acc_l1<<VEC_FROM_ARRAY(tail->acc);
          angvel_l1<<VEC_FROM_ARRAY(tail->gyr);

          for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--)
          {
            dt_l1 = it_pcl->curvature / double(1000) - head->offset_time;
            M3D R_i(R_l1 * Exp(angvel_l1, dt_l1));
            V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
            V3D T_ei(pos_l1 + vel_l1 * dt_l1 + 0.5 * acc_l1 * dt_l1 * dt_l1 - imu_state.pos);
            V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);

            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);

            if (it_pcl == pcl_L1_out->points.begin()) break;
          }
        }
      }
    }

    #pragma omp section
    {
      if (l2_own_imu && !pcl_L2_out->points.empty())
      {
        l2_snapshot_.applyTo(*pcl_L2_out);
      }
      else if (!pcl_L2_out->points.empty())
      {
        V3D angvel_l2, acc_l2, vel_l2, pos_l2;
        M3D R_l2;
        double dt_l2 = 0;
        auto it_pcl = pcl_L2_out->points.end() - 1;
        for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
        {
          auto head = it_kp - 1;
          auto tail = it_kp;
          R_l2<<MAT_FROM_ARRAY(head->rot);
          vel_l2<<VEC_FROM_ARRAY(head->vel);
          pos_l2<<VEC_FROM_ARRAY(head->pos);
          acc_l2<<VEC_FROM_ARRAY(tail->acc);
          angvel_l2<<VEC_FROM_ARRAY(tail->gyr);

          for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--)
          {
            dt_l2 = it_pcl->curvature / double(1000) - head->offset_time;
            M3D R_i(R_l2 * Exp(angvel_l2, dt_l2));
            V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
            V3D T_ei(pos_l2 + vel_l2 * dt_l2 + 0.5 * acc_l2 * dt_l2 * dt_l2 - imu_state.pos);
            V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);

            it_pcl->x = P_compensate(0);
            it_pcl->y = P_compensate(1);
            it_pcl->z = P_compensate(2);

            if (it_pcl == pcl_L2_out->points.begin()) break;
          }
        }
      }
    }
  }

  if (l2_own_imu)
  {
    /*** merge an L1-framed copy for the EKF; pcl_L2_out itself stays in the native L2 frame ***/
    const auto &ext = l2_undistort_->extrinsics();
    pcl_in_out = *pcl_L1_out;
    pcl_in_out.points.reserve(pcl_in_out.points.size() + pcl_L2_out->points.size());
    for (const auto& pt : pcl_L2_out->points)
    {
      PointType p = pt;
      V3D p_L1 = ext.lidar2_R_wrt_L1 * V3D(pt.x, pt.y, pt.z) + ext.lidar2_T_wrt_L1;
      p.x = p_L1(0); p.y = p_L1(1); p.z = p_L1(2);
      pcl_in_out.push_back(p);
    }
  }
  else
  {
    pcl_in_out = *pcl_L1_out + *pcl_L2_out;
    if (try_l2_own_imu)
    {
      /*** keep the per-lidar output in the native L2 frame (undo the fallback transform) ***/
      const auto &ext = l2_undistort_->extrinsics();
      for (auto& pt : pcl_L2_out->points)
      {
        V3D p_L2 = ext.lidar2_R_wrt_L1.transpose() * (V3D(pt.x, pt.y, pt.z) - ext.lidar2_T_wrt_L1);
        pt.x = p_L2(0); pt.y = p_L2(1); pt.z = p_L2(2);
      }
    }
  }
  last_undistort_state_ = imu_state;
}

void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, const PointCloudXYZI::Ptr& pcl_un_, const PointCloudXYZI::Ptr& pcl_L1_out, const PointCloudXYZI::Ptr& pcl_L2_out, const bool &multi_lidar)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();

  pcl_un_->clear();
  if (multi_lidar) { pcl_L1_out->clear(); pcl_L2_out->clear(); }
  l2_snapshot_valid_ = false;
  // Gyro-bias init for the secondary IMU. Runs beyond the (short) filter init phase
  // until enough samples arrived — L2 scans may not appear during filter init at all.
  if (l2_undistort_ && !l2_undistort_->biasReady()) l2_undistort_->feedInit(meas.imu2);

  if (meas.imu.empty() && imu_need_init_) { return; }
  if (meas.lidar == nullptr) {
    std::cerr << "[ImuProcess] lidar point cloud is null, skipping frame\n";
    return;
  }

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      std::cout << "IMU Initial Done" << std::endl;
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }

  if (multi_lidar) UndistortPclMultiLiDAR(meas, kf_state, *pcl_un_, pcl_L1_out, pcl_L2_out);
  else UndistortPcl(meas, kf_state, *pcl_un_, pcl_L2_out);

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
