cmake_minimum_required(VERSION 2.8.3)
project(vins_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  message_runtime
  geometry_msgs
  roscpp
  sensor_msgs
  std_msgs
)

add_message_files(
    FILES
    preintegration_msg.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)
