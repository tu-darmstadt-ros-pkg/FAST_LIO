std_msgs/Header header
geometry_msgs/Accel peak_accel      # per-axis peak: linear [m/s²], angular [rad/s] (lidar frame, gravity-removed)
float64 lin_mag_avg                 # mean |a_true| across IMU samples in scan [m/s²]
float64 lin_mag_max                 # peak |a_true| across IMU samples in scan [m/s²]
float64 omega_mag_avg               # mean |ω| across IMU samples in scan [rad/s]
float64 omega_mag_max               # peak |ω| across IMU samples in scan [rad/s]
float64 slerp_score                 # combined motion quality score [m]
