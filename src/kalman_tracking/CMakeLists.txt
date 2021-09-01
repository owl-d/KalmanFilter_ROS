cmake_minimum_required(VERSION 3.0.2)
project(kalman_tracking)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  mavros
  mavros_msgs
  mavros_extras
  message_generation
)

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
#   mavros_msgs
#   rospy
# )

catkin_package(
  #  INCLUDE_DIRS include
  #  LIBRARIES kalman_tracking
  # CATKIN_DEPENDS mavros mavros_extras rospy std_msgs
 ## DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)