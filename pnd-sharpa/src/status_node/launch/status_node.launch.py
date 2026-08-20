"""Launch the Joy-driven teleop status node."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="status_node",
                executable="status",
                name="status",
                output="screen",
                parameters=[
                    {
                        "status_topic": "/control_status",
                        "status_json_topic": "/teleop/status_json",
                        "joy_topics": ["/xbox/joy"],
                        "device": "/dev/input/js0",
                        "device_backend": "auto",
                        "read_device": True,
                        "publish_joy": True,
                        "joy_output_topic": "/joy",
                        "axis_count": 8,
                        "button_count": 15,
                        "poll_period": 0.005,
                        "dpad_x_axis": 6,
                        "dpad_y_axis": 7,
                        "dpad_y_up_sign": -1,
                        "lb_button": 6,
                        "b_button": 1,
                        "lt_axis": 5,
                        "lt_button": 8,
                        "lt_axis_threshold": 0.5,
                        "lt_pressed_when": "above",
                        "rt_axis": 4,
                        "rt_button": 9,
                        "rt_axis_threshold": 0.5,
                        "rt_pressed_when": "above",
                    }
                ],
            )
        ]
    )
