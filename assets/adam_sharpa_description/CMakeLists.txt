cmake_minimum_required(VERSION 3.8)
project(adam_sharpa_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

install(DIRECTORY mujoco/
  DESTINATION share/${PROJECT_NAME}/mujoco
)

ament_package()
