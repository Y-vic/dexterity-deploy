cmake_minimum_required(VERSION 3.8)
project(noitom_node)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  vendor/bin/noitom_mocap
  DESTINATION lib/${PROJECT_NAME}
  RENAME noitom_mocap
)

install(PROGRAMS
  vendor/bin/adam_retarget
  DESTINATION lib/${PROJECT_NAME}
  RENAME noitom_retarget
)

install(PROGRAMS
  scripts/noitom_gate.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME noitom
)

install(PROGRAMS
  scripts/noitom_gmr_retarget.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME noitom_gmr_retarget
)

install(PROGRAMS
  scripts/noitom_mink_retarget.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME noitom_mink_retarget
)

install(PROGRAMS
  scripts/noitom_retarget_evaluator.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME noitom_retarget_evaluator
)

install(DIRECTORY
  vendor/lib
  vendor/share
  DESTINATION share/${PROJECT_NAME}/vendor
)

ament_package()
