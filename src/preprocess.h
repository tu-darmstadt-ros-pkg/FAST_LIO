// #include <ros/ros.h>
#include <rclcpp/rclcpp.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>

using namespace std;

#define IS_VALID(a) ((abs(a) > 1e8) ? true : false)

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

enum LID_TYPE
{
  LIVOX_CUSTOM = 1,  // livox_ros_driver2::msg::CustomMsg
  VELODYNE,          // velodyne_ros::Point (x,y,z,intensity,time,ring)
  OUSTER,            // ouster_ros::Point (x,y,z,intensity,t,reflectivity,ring,ambient,range)
  XYZRTL,            // livox_ros::LivoxPointXyzrtl (x,y,z,reflectivity,tag,line — no per-point time)
  XYZRTLO_AVIA,      // x,y,z,intensity,tag,ring,time (configurable field names) + Avia scan-line grouping & duplicate filtering
  XYZRTLO            // x,y,z,intensity,tag,ring,time (configurable field names), simple (t used directly)
};
enum TIME_UNIT
{
  SEC = 0,
  MS = 1,
  US = 2,
  NS = 3
};
enum Feature
{
  Nor,
  Poss_Plane,
  Real_Plane,
  Edge_Jump,
  Edge_Plane,
  Wire,
  ZeroPoint
};
enum Surround
{
  Prev,
  Next
};
enum E_jump
{
  Nr_nor,
  Nr_zero,
  Nr_180,
  Nr_inf,
  Nr_blind
};

struct orgtype
{
  double range;
  double dista;
  double angle[2];
  double intersect;
  E_jump edj[2];
  Feature ftype;
  orgtype()
  {
    range = 0;
    edj[Prev] = Nr_nor;
    edj[Next] = Nr_nor;
    ftype = Nor;
    intersect = 2;
  }
};

namespace velodyne_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D;
  float intensity;
  float time;
  uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity,
                                                                          intensity)(float, time, time)(uint16_t, ring,
                                                                                                        ring))

namespace ouster_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D;
  float intensity;
  uint32_t t;
  uint16_t reflectivity;
  uint16_t ring;
  uint16_t ambient;
  uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint16_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)

namespace livox_ros
{
typedef struct {
  float x;            /**< X axis, Unit:m */
  float y;            /**< Y axis, Unit:m */
  float z;            /**< Z axis, Unit:m */
  float reflectivity; /**< Reflectivity   */
  uint8_t tag;        /**< Livox point tag   */
  uint8_t line;       /**< Laser line id     */
} LivoxPointXyzrtl;
}
POINT_CLOUD_REGISTER_POINT_STRUCT(livox_ros::LivoxPointXyzrtl,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, reflectivity, reflectivity)
    (uint8_t, tag, tag)
    (uint8_t, line, line)
)

class Preprocess
{
  public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Preprocess();
  ~Preprocess();
  
  void process(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void process(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void set(bool feat_en, int lid_type, double bld, int pfilt_num);

  // sensor_msgs::PointCloud2::ConstPtr pointcloud;
  PointCloudXYZI pl_full, pl_corn, pl_surf;
  PointCloudXYZI pl_buff[128]; //maximum 128 line lidar
  vector<orgtype> typess[128]; //maximum 128 line lidar
  float time_unit_scale;
  int lidar_type, point_filter_num, N_SCANS, SCAN_RATE, time_unit;
  double blind;
  bool feature_enabled, given_offset_time, self_filtered;
  // ros::Publisher pub_full, pub_surf, pub_corn;

  // Configurable PointCloud2 field names for the XYZRTLO / XYZRTLO_AVIA handlers, since different
  // Livox PointCloud2 drivers/configs may publish these under different field names
  // (e.g. custom_pc2_intensity_field/custom_pc2_tag_field/custom_pc2_ring_field/custom_pc2_time_field
  // in athena_livox_driver.yaml).
  std::string field_name_intensity = "intensity";
  std::string field_name_tag = "tag";
  std::string field_name_ring = "ring";
  std::string field_name_time = "t";

private:
  struct PointFieldInfo
  {
    int offset = -1;
    uint8_t datatype = 0;
  };
  struct XyzrtloFieldLayout
  {
    PointFieldInfo intensity, tag, ring, time;
    bool valid = false;
  };

  XyzrtloFieldLayout lookupXyzrtloFieldLayout(const sensor_msgs::msg::PointCloud2 &msg);
  static PointFieldInfo findField(const sensor_msgs::msg::PointCloud2 &msg, const std::string &name);
  template <typename T>
  static T readFieldValue(const uint8_t *point_data, const PointFieldInfo &info);

  void livox_custom_handler(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg);
  void xyzrtlo_avia_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
  void ouster_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
  void velodyne_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
  void xyzrtl_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
  void xyzrtlo_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
  void default_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
  void give_feature(PointCloudXYZI &pl, vector<orgtype> &types);
  void pub_func(PointCloudXYZI &pl, const rclcpp::Time &ct);
  int  plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool small_plane(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir);
  
  int group_size;
  double disA, disB, inf_bound;
  double limit_maxmid, limit_midmin, limit_maxmin;
  double p2l_ratio;
  double jump_up_limit, jump_down_limit;
  double cos160;
  double edgea, edgeb;
  double smallp_intersect, smallp_ratio;
  double vx, vy, vz;
};
