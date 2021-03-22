std_msgs/Header           header

geometry_msgs/Vector3[]   lin_acc
geometry_msgs/Vector3[]   ang_vel
float32[]                 dt
float32[3]                acc0
float32[3]                gyr0
float32[3]                bas
float32[3]                bgs
float32[3]                vs
#geometry_msgs/Vector3   acc0
#geometry_msgs/Vector3   gyr0
#geometry_msgs/Vector3   bas
#geometry_msgs/Vector3   bgs
#geometry_msgs/Vector3   vs

#params
float32                  sigma_ac
float32                  sigma_gc
float32                  sigma_awc
float32                  sigma_gwc
float32                  g
float32                  td
