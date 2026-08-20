cmake_minimum_required(VERSION 3.8)
project(teleop_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TactileDeformImage.msg"
  "msg/TactileDeformImageArray.msg"
  "msg/TactileForce6D.msg"
  "msg/TactileForce6DArray.msg"
  "msg/TactileContactPoints.msg"
  "msg/TactileContactPointsArray.msg"
  "msg/SharpaJointState.msg"
  DEPENDENCIES
    std_msgs
    builtin_interfaces
    sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
