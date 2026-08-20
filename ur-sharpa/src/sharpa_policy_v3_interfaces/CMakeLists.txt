cmake_minimum_required(VERSION 3.10)
project(sharpa_policy_v3_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CameraFrame.msg"
  "msg/RobotStateFrame.msg"
  "msg/TauFrame.msg"
  "msg/WrenchFrame.msg"
  "msg/DeformationFrame.msg"
  "msg/UrStateFrame.msg"
  "msg/HandStateFrame.msg"
  "msg/ArmCommand.msg"
  "msg/HandCommand.msg"
  "msg/HardwareCommandResult.msg"
  "msg/PolicyActionV3.msg"
  "msg/ExecutionFeedback.msg"
  "msg/PolicyFault.msg"
  "srv/ResetPolicy.srv"
  "srv/BuildObservation.srv"
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
