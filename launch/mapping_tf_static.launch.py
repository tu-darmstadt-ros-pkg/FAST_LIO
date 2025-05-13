from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch args
    config_file = LaunchConfiguration('config_file')
    static_parent_frame_id = LaunchConfiguration('static_parent_frame_id')
    static_child_frame_id = LaunchConfiguration('static_child_frame_id')

    # Declare launch args
    declare_config_file = DeclareLaunchArgument(
        'config_file',
        description='Full path to the YAML config file for fast_lio'
    )
    declare_base_frame_id = DeclareLaunchArgument(
        'base_frame_id',
        default_value='map',
        description='Parent frame of the static transform'
    )
    declare_map_frame_id = DeclareLaunchArgument(
        'map_frame_id',
        default_value='base_link',
        description='Child frame of the static transform'
    )

    # fastlio_mapping node
    fastlio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        name='fast_lio',
        output='screen',
        parameters=[config_file],
    )

    # Static transform publisher (0 0 0 0 0 0)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='fastlio_static_tf',
        arguments=['0', '0', '0', '0', '0', '0',
                   static_parent_frame_id, static_child_frame_id],
        output='screen'
    )

    return LaunchDescription([
        declare_config_file,
        declare_base_frame_id,
        declare_map_frame_id,
        fastlio_node,
        static_tf
    ])
