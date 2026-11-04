from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    arguments=[DeclareLaunchArgument("recording_root",default_value="/media/yns/T9/0824_yns_unscrew_cap"),DeclareLaunchArgument("video_ffmpeg",default_value="ffmpeg"),
               DeclareLaunchArgument("skip_tactile", default_value="false")]
    parameters={key:LaunchConfiguration(key) for key in ("recording_root","video_ffmpeg")}; parameters["frequency"]=30.0
    parameters['skip_tactile'] = ParameterValue(LaunchConfiguration('skip_tactile'), value_type=bool)
    return LaunchDescription(arguments+[Node(package="recording_node",executable="recording_node",name="recording_node",output="screen",parameters=[parameters])])
