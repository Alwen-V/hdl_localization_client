#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <hdl_localization/TMMAction.h>  //这是action文件编译出来的，很重要！！
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>
#include <Eigen/Dense>
#include <geometry_msgs/PoseStamped.h>
#include <hdl_localization/TMMActionResult.h>
#include <hdl_localization/TMMActionFeedback.h>
#include <hdl_localization/TMMActionGoal.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
typedef actionlib::SimpleActionClient<hdl_localization::TMMAction> action_Client;
struct PointXYZIRPYT {
  PCL_ADD_POINT4D;    // preferred way of adding a XYZ+padding
  PCL_ADD_INTENSITY;  // add intensity
  double roll;
  double pitch;
  double yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // ensure proper alignment
} EIGEN_ALIGN16;

// Register the point type with PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(
  PointXYZIRPYT,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(double, roll, roll)(double, pitch, pitch)(double, yaw, yaw)(double, time, time))
typedef PointXYZIRPYT PointTypePose;

class TMMClient {
public:
  TMMClient(ros::NodeHandle& nh) : nh_(nh), ac_("tmm_action", true), temporaryCloudKeyPoses6D(new pcl::PointCloud<PointTypePose>) {
    ac_.waitForServer();
    ROS_INFO("111----------------Action client started, sending goal.---------------");
    // 重置temporaryCloudKeyPoses6D、point_clouds_body
    point_clouds_body.clear();
  }
  void sendPose(const geometry_msgs::PoseStamped& pose, bool trigger);

public:
  ros::NodeHandle nh_;
  action_Client ac_;
  // ros::Publisher latest_odom_pub_;
  // nav_msgs::Odometry current_odom_;
  sensor_msgs::PointCloud2 current_point_cloud_;
  // const double inlier_threshold_ = 0.5;
  geometry_msgs::TransformStamped feedback_odom_trans;
  nav_msgs::Odometry feedback_odomAftMapped;
  sensor_msgs::PointCloud2 frame_point_cloud;
  pcl::PointCloud<PointTypePose>::Ptr temporaryCloudKeyPoses6D;
  std::vector<boost::shared_ptr<pcl::PointCloud<pcl::PointXYZI>>> point_clouds_body;

public:
  void activeCb() { ROS_INFO("333Goal just went active"); }

  void feedbackCb(const hdl_localization::TMMFeedbackConstPtr& feedback)  // 来自fastlio连续的action反馈
  {
    // ROS_INFO("Got Feedback - alignment_score: %f, inlier_ratio: %f", feedback->alignment_score, feedback->inlier_ratio);
    // ROS_INFO(" 666~~~~~~~~~~Run feedbackCb~~~~~~~~~~ ");

    //!!这里应该传入的是fastlio的带时间戳的geometry_msgs::TransformStamped类型
    feedback_odom_trans = feedback->odom_trans;
    feedback_odomAftMapped = feedback->odomAftMapped;
    frame_point_cloud = feedback->frame_point_cloud;
  }

  // doneCb 回调函数：在 Action 完成后处理结果。 定义一个将feed_back的result的位姿转换为temporaryCloudKeyPoses6D的函数
  void doneCb(const actionlib::SimpleClientGoalState& state, const hdl_localization::TMMResultConstPtr& result) {
    ROS_INFO("Action finished: %s", state.toString().c_str());
    std::cout << "~~~~~~~~~~~~" << "result_.temporary_odometry.odometries.size():" << result->temporary_odometry.odometries.size() << "~~~~~~~~~~~~" << std::endl;
    // Handle the result here
    for (const auto& odom : result->temporary_odometry.odometries) {
      PointXYZIRPYT point;
      point.x = odom.pose.pose.position.x;
      point.y = odom.pose.pose.position.y;
      point.z = odom.pose.pose.position.z;
      point.intensity = 1.0;  // 可以设置为特定值,用来辨别那些位姿点对应的地图点是临时建图的得到的

      // Convert orientation (quaternion) to roll, pitch, and yaw
      tf::Quaternion q(odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w);
      tf::Matrix3x3 m(q);
      m.getRPY(point.roll, point.pitch, point.yaw);

      point.time = odom.header.stamp.toSec();

      temporaryCloudKeyPoses6D->push_back(point);

      ROS_INFO("Odometry: position(%f, %f, %f), roll(%f), pitch(%f), yaw(%f)", point.x, point.y, point.z, point.roll, point.pitch, point.yaw);

      // 处理result的temporary_point_cloud
      for (const auto& pointcloud_msg : result->temporary_point_cloud.point_clouds) {
        boost::shared_ptr<pcl::PointCloud<pcl::PointXYZI>> pcl_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        // 将sensor_msgs::PointCloud2转换为PCL点云
        pcl::fromROSMsg(pointcloud_msg, *pcl_cloud);
        //! 这里可以标记临时建图的点云, 比如设置intensity为1.0
        for (auto& p : pcl_cloud->points) {
          p.intensity = 1.0;
        }
        point_clouds_body.push_back(pcl_cloud);
      }
    }

    //! 这里应该完成恢复hdl的状态
  }

  bool isDone() { return ac_.getState().isDone(); }
  void cancelGoal() { ac_.cancelGoal(); }
};

void TMMClient::sendPose(const geometry_msgs::PoseStamped& pose, bool trigger) {
  hdl_localization::TMMGoal goal;
  goal.trigger = trigger;
  goal.current_pose = pose;  // 这里添加你最新的位姿

  ac_.sendGoal(goal, boost::bind(&TMMClient::doneCb, this, _1, _2), boost::bind(&TMMClient::activeCb, this), boost::bind(&TMMClient::feedbackCb, this, _1));

  ROS_INFO("555Goal sent, waiting for result.");
}
