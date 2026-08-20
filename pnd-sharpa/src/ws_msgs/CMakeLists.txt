cmake_minimum_required(VERSION 3.8)
project(ws_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RobotState.msg"
  "msg/RobotTactile.msg"
  "msg/ModelImage.msg"
  "msg/PolicyObs.msg"
  "msg/PolicyPred.msg"
  "msg/InferenceRequest.msg"
  "msg/ActionPlan.msg"
  "msg/ExecutionDone.msg"
  "msg/PndAction.msg"
  "msg/Status.msg"
  DEPENDENCIES
    std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
