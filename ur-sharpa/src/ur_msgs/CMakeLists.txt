cmake_minimum_required(VERSION 3.8)
project(ur_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  # hardware layer
  "msg/CameraSet.msg"
  "msg/ArmCommand.msg"
  "msg/UrStateFrame.msg"
  "msg/HandCommand.msg"
  "msg/HandStateFrame.msg"
  "msg/TauFrame.msg"
  "msg/WrenchFrame.msg"
  "msg/DeformationFrame.msg"
  # control layer
  "msg/ControlStatus.msg"
  "msg/HardwareCommandResult.msg"
  # deploy layer
  "msg/PolicyObs.msg"
  "msg/ExecutableAction.msg"
  "msg/ActorTrajectory.msg"
  "msg/ExecutionDone.msg"
  # services
  "srv/BuildObservation.srv"
  "srv/ResetPolicy.srv"
  DEPENDENCIES
    std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
