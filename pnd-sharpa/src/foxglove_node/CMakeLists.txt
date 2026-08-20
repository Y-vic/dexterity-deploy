cmake_minimum_required(VERSION 3.8)
project(foxglove_node)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/foxglove_joint_state_merge.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME foxglove_joint_state_merge
)

install(FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
