#include <mutex>
#include <memory>
#include <iostream>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <eigen_conversions/eigen_msg.h>

#include <std_srvs/Empty.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>

#include <pclomp/ndt_omp.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>

#include <hdl_localization/pose_estimator.hpp>
#include <hdl_localization/delta_estimater.hpp>

#include <hdl_localization/ScanMatchingStatus.h>
#include <hdl_global_localization/SetGlobalMap.h>
#include <hdl_global_localization/QueryGlobalLocalization.h>
// #include "/home/alwen/project_learning/FAST-LIO_ws/src/hdl_localization/include/hdl_localization/hdl_action_client.hpp"
#include "hdl_localization/hdl_action_client.hpp"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <future>
#include <memory>
#include <chrono>
#include <string>

#include <cmath>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include <boost/filesystem.hpp>

namespace hdl_localization {

struct VoxelGrid {
  float voxel_size;  // 0.5
  int grid_length;
  int grid_width;
  int grid_height;
  Eigen::Vector3f origin;  // 代表体素中心，也是map系原点。

  VoxelGrid(float size, int length, int width, int height) : voxel_size(size), grid_length(length), grid_width(width), grid_height(height) {
    origin = Eigen::Vector3f(grid_length * voxel_size / 2.0f, grid_width * voxel_size / 2.0f, grid_height * voxel_size / 2.0f);
  }

  int getVoxelIndex(float x, float y, float z) const {
    int ix = static_cast<int>(std::floor((x + origin.x()) / voxel_size));
    int iy = static_cast<int>(std::floor((y + origin.y()) / voxel_size));
    int iz = static_cast<int>(std::floor((z + origin.z()) / voxel_size));
    return (ix * grid_width * grid_height) + (iy * grid_height) + iz;
  }

  Eigen::Vector3i getVoxelCenter(int index) const {
    int ix = index / (grid_width * grid_height);
    int iy = (index % (grid_width * grid_height)) / grid_height;
    int iz = index % grid_height;
    //   return Eigen::Vector3f(
    //     (ix * voxel_size) - origin.x() + voxel_size / 2.0f,
    //     (iy * voxel_size) - origin.y() + voxel_size / 2.0f,
    //     (iz * voxel_size) - origin.z() + voxel_size / 2.0f);
    // }
    return Eigen::Vector3i(ix, iy, iz);
  }
};

struct PointCloudVoxelInfo {
  pcl::PointXYZ global_pose;
  std::unordered_set<int> occupied_voxel_indices;
  std::unordered_set<int> unoccupied_voxels_indices;
  std::vector<Eigen::Vector3i> voxelCenters;  //

  PointCloudVoxelInfo(float x, float y, float z) : global_pose(x, y, z) {}
};

class HdlLocalizationNodelet : public nodelet::Nodelet {
public:
  using PointT = pcl::PointXYZI;

  HdlLocalizationNodelet() : tf_buffer(), tf_listener(tf_buffer), timeout_duration_(5.0) {}
  virtual ~HdlLocalizationNodelet() {}

  void onInit() override {
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();
    TMM_end_priorPose.resize(100, Eigen::Matrix4f::Identity());

    initialize_params();

    robot_odom_frame_id = private_nh.param<std::string>("robot_odom_frame_id", "robot_odom");  // odom
    odom_child_frame_id = private_nh.param<std::string>("odom_child_frame_id", "base_link");   // velodyne

    use_imu = private_nh.param<bool>("use_imu", true);
    invert_acc = private_nh.param<bool>("invert_acc", false);
    invert_gyro = private_nh.param<bool>("invert_gyro", false);
    if (use_imu) {
      NODELET_INFO("enable imu-based prediction");
      imu_sub = mt_nh.subscribe("/gpsimu_driver/imu_data", 256, &HdlLocalizationNodelet::imu_callback, this);
    }
    points_sub = mt_nh.subscribe("/velodyne_points", 5, &HdlLocalizationNodelet::points_callback, this);
    globalmap_sub = nh.subscribe("/globalmap", 1, &HdlLocalizationNodelet::globalmap_callback, this);
    initialpose_sub = nh.subscribe("/initialpose", 8, &HdlLocalizationNodelet::initialpose_callback, this);

    // odometry_sub = nh.subscribe("/odom", 1, &HdlLocalizationNodelet::odometryCallback, this);

    pose_pub = nh.advertise<nav_msgs::Odometry>("/odom", 5, false);
    aligned_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_points", 5, false);
    status_pub = nh.advertise<ScanMatchingStatus>("/status", 5, false);
    map_update_pub = nh.advertise<std_msgs::String>("/map_request/pcd", 5, false);

    // global localization
    use_global_localization = private_nh.param<bool>("use_global_localization", true);
    if (use_global_localization) {
      NODELET_INFO_STREAM("wait for global localization services");
      ros::service::waitForService("/hdl_global_localization/set_global_map");
      ros::service::waitForService("/hdl_global_localization/query");

      set_global_map_service = nh.serviceClient<hdl_global_localization::SetGlobalMap>("/hdl_global_localization/set_global_map");
      query_global_localization_service = nh.serviceClient<hdl_global_localization::QueryGlobalLocalization>("/hdl_global_localization/query");

      relocalize_server = nh.advertiseService("/relocalize", &HdlLocalizationNodelet::relocalize, this);
    }
    // Add subscriber for ScanMatchingStatus
    status_sub = nh.subscribe("/status", 5, &HdlLocalizationNodelet::status_callback, this);

    // Initialize inlier threshold
    inlier_threshold_ = private_nh.param<double>("inlier_threshold", 0.94);

    // 创建一个定时器来检查bag包是否已经播放完毕
    // timer_ = nh.createTimer(ros::Duration(1.0), &HdlLocalizationNodelet::timerCallback, this);
    timer_ = nh.createWallTimer(ros::WallDuration(1.0), &HdlLocalizationNodelet::timerCallback, this);
  }

private:
  pcl::Registration<PointT, PointT>::Ptr create_registration() const {
    std::string reg_method = private_nh.param<std::string>("reg_method", "NDT_OMP");
    std::string ndt_neighbor_search_method = private_nh.param<std::string>("ndt_neighbor_search_method", "DIRECT7");
    double ndt_neighbor_search_radius = private_nh.param<double>("ndt_neighbor_search_radius", 2.0);
    double ndt_resolution = private_nh.param<double>("ndt_resolution", 1.0);

    if (reg_method == "NDT_OMP") {
      NODELET_INFO("NDT_OMP is selected");
      pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
      ndt->setTransformationEpsilon(0.01);
      ndt->setResolution(ndt_resolution);
      if (ndt_neighbor_search_method == "DIRECT1") {
        NODELET_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        NODELET_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
      } else {
        if (ndt_neighbor_search_method == "KDTREE") {
          NODELET_INFO("search_method KDTREE is selected");
        } else {
          NODELET_WARN("invalid search method was given");
          NODELET_WARN("default method is selected (KDTREE)");
        }
        ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
      }
      return ndt;
    } else if (reg_method.find("NDT_CUDA") != std::string::npos) {
      NODELET_INFO("NDT_CUDA is selected");
      boost::shared_ptr<fast_gicp::NDTCuda<PointT, PointT>> ndt(new fast_gicp::NDTCuda<PointT, PointT>);
      ndt->setResolution(ndt_resolution);

      if (reg_method.find("D2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::D2D);
      } else if (reg_method.find("P2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
      }

      if (ndt_neighbor_search_method == "DIRECT1") {
        NODELET_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        NODELET_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
      } else if (ndt_neighbor_search_method == "DIRECT_RADIUS") {
        NODELET_INFO_STREAM("search_method DIRECT_RADIUS is selected : " << ndt_neighbor_search_radius);
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT_RADIUS, ndt_neighbor_search_radius);
      } else {
        NODELET_WARN("invalid search method was given");
      }
      return ndt;
    }

    NODELET_ERROR_STREAM("unknown registration method:" << reg_method);
    return nullptr;
  }

  void initialize_params() {
    // intialize scan matching method
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    downsample_filter = voxelgrid;

    NODELET_INFO("create registration method for localization");
    registration = create_registration();

    // global localization
    NODELET_INFO("create registration method for fallback during relocalization");
    relocalizing = false;
    delta_estimater.reset(new DeltaEstimater(create_registration()));

    // initialize pose estimator
    if (private_nh.param<bool>("specify_init_pose", true)) {
      NODELET_INFO("initialize pose estimator with specified parameters!!");
      pose_estimator.reset(new hdl_localization::PoseEstimator(
        registration,
        Eigen::Vector3f(private_nh.param<double>("init_pos_x", 0.0), private_nh.param<double>("init_pos_y", 0.0), private_nh.param<double>("init_pos_z", 0.0)),
        Eigen::Quaternionf(
          private_nh.param<double>("init_ori_w", 1.0),
          private_nh.param<double>("init_ori_x", 0.0),
          private_nh.param<double>("init_ori_y", 0.0),
          private_nh.param<double>("init_ori_z", 0.0)),
        private_nh.param<double>("cool_time_duration", 0.5)));
    }

    // 定义一个全局固定的地图
    //  read globalmap from a pcd file
    std::string globalmap_pcd_stable = private_nh.param<std::string>("globalmap_pcd", "");
    globalmap_stable.reset(new pcl::PointCloud<PointT>());
    pcl::io::loadPCDFile(globalmap_pcd_stable, *globalmap_stable);
    globalmap_stable->header.frame_id = "map";
    voxelgrid->setInputCloud(globalmap_stable);
    voxelgrid->filter(*globalmap_stable);
  }

  void readOdometryFile(const std::string& filename, pcl::PointCloud<PointTypePose>::Ptr cloud) {
    std::ifstream infile(filename);

    if (!infile.is_open()) {
      std::cerr << "Unable to open file: " << filename << std::endl;
    }

    std::string line;
    while (std::getline(infile, line)) {
      std::istringstream iss(line);
      PointTypePose point;
      float x, y, z, qx, qy, qz, qw, time;

      // if (!(iss >> time  >> x >> y >> z >> qx >> qy >> qz >> qw)) {
      //   std::cerr << "Error parsing line: " << line << std::endl;
      //   continue;
      // }
      // point.x = x;
      // point.y = y;
      // point.z = z;
      // point.intensity = 1.0;  // Assuming intensity is set to 1.0
      // tf::Quaternion q(qx, qy, qz, qw);
      std::string token;

      if (std::getline(iss, token, ',')) point.time = std::stof(token);
      if (std::getline(iss, token, ',')) point.x = std::stof(token);
      if (std::getline(iss, token, ',')) point.y = std::stof(token);
      if (std::getline(iss, token, ',')) point.z = std::stof(token);
      if (std::getline(iss, token, ',')) qx = std::stof(token);
      if (std::getline(iss, token, ',')) qy = std::stof(token);
      if (std::getline(iss, token, ',')) qz = std::stof(token);
      if (std::getline(iss, token, ',')) qw = std::stof(token);
      point.intensity = 1.0;
      tf::Quaternion q(qx, qy, qz, qw);
      tf::Matrix3x3 m(q);
      m.getRPY(point.roll, point.pitch, point.yaw);

      point.time = time;

      // 将点添加到点云中
      cloud->push_back(point);
    }

    infile.close();
  }

private:
  /**
   * @brief callback for imu data
   * @param imu_msg
   */
  void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
    std::lock_guard<std::mutex> lock(imu_data_mutex);
    imu_data.push_back(imu_msg);
  }

  // void timerCallback(const ros::TimerEvent&) const ros::WallTimerEvent& event
  void timerCallback(const ros::WallTimerEvent& event) {
    std::cout << "~~~~~~~~~timerCallback~~~~~~~~~" << std::endl;
    ROS_INFO_STREAM("Current ROS time: " << ros::WallTime::now());
    if ((ros::WallTime::now() - last_received_time).toSec() > timeout_duration_) {
      if (received_points_flag) {
        ROS_INFO("No new points received, bag playback might be finished.");
        received_points_flag = false;
        // 记录此时fastlio反馈的里程计位姿，作为因子图优化的位姿先验。
        float x = action_client->feedback_odomAftMapped.pose.pose.position.x;
        float y = action_client->feedback_odomAftMapped.pose.pose.position.y;
        float z = action_client->feedback_odomAftMapped.pose.pose.position.z;

        // 提取方向（四元数）
        float qx = action_client->feedback_odomAftMapped.pose.pose.orientation.x;
        float qy = action_client->feedback_odomAftMapped.pose.pose.orientation.y;
        float qz = action_client->feedback_odomAftMapped.pose.pose.orientation.z;
        float qw = action_client->feedback_odomAftMapped.pose.pose.orientation.w;

        Eigen::Quaternionf quat(qw, qx, qy, qz);
        Eigen::Matrix3f rotation_matrix = quat.toRotationMatrix();
        Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
        init_guess.block<3, 3>(0, 0) = rotation_matrix;
        init_guess(0, 3) = x;
        init_guess(1, 3) = y;
        init_guess(2, 3) = z;
        pcl::PointCloud<pcl::PointXYZI>::Ptr lastPointCloud(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::fromROSMsg(action_client->frame_point_cloud, *lastPointCloud);
        registration->setInputSource(lastPointCloud);
        registration->align(*lastPointCloud, init_guess);

        Eigen::Matrix4f trans = registration->getFinalTransformation();

        TMM_end_priorPose[count_needMergeMap + 1] = registration->getFinalTransformation();  //! 最后需要停在先验地图范围内,且需要注意TMM_end_priorPose是从1开始记录有效数据的
        std::string TMM_end_prior_pose = TMM_saveDir + "/TMM_end_prior_pose.txt";
        saveMatricesToFile(TMM_end_priorPose, TMM_end_prior_pose);

        NODELET_INFO("endPriorPose caculated at timer_callback()!!");
      }
    }
    if ((action_client && !action_client->isDone() && TMM_cancel_flag >= 10) || (!received_points_flag && action_client)) {
      std::cout << "~~~~~~~~~已经满足连续9次内点阈值条件,可以执行合并地图~~~~~~~~~" << std::endl;
      //
      action_client->cancelGoal();
      std::cout << "~~~~~~~~~ action_client->cancelGoal();~~~~~~~~~" << std::endl;
      // ros::WallDuration(0.4).sleep();
      // action_client.reset();  // 调用析构,释放内存,指针重置  不能放这里

      TMM_cancel_flag = 0;
      ROS_INFO("Inlier ratio above threshold 5 times, cancel the goal.");
      goodToMergeMap = true;

      count_needMergeMap++;
      action_client.reset();
    }
    nh.getParam("/merge_maps", merge_maps_);
    nh.getParam("TMM_saveDir", TMM_saveDir);
    // 如果可以合并地图
    if (goodToMergeMap && merge_maps_) {  // 可以合并地图或者播包结束，则合并地图   。
      merge_maps_ = false;
      goodToMergeMap = false;
      for (int i = 1; i <= count_needMergeMap; i++) {
        // std::cout << "~~~~~~~~~ action_client->temporaryCloudKeyPoses6D~~~~~~~~~" << action_client->temporaryCloudKeyPoses6D->size() << std::endl;
        pcl::PointCloud<PointTypePose>::Ptr TMModom(new pcl::PointCloud<PointTypePose>());
        std::string TMM_each_path = TMM_saveDir + "/TMM_saveDir_" + std::to_string(i);
        std::string TMM_each_path_odom = TMM_each_path + "/TMM_odom.txt";
        readOdometryFile(TMM_each_path_odom, TMModom);

        if (!TMModom->empty()) {
          std::cout << "~~~~~~~~~开始执行合并区域" << i << "的地图~~~~~~~~~" << TMModom->points.size() << std::endl;
          nh.getParam("filename_Merge_result_", filename_Merge_result_);
          mergeMap(globalmap_stable, TMModom, i);
        } else {
          std::cout << "~~~~~~~~~temporaryCloudKeyPoses6D is null~~~~~~~~~" << std::endl;
        }

        // if (action_client->temporaryCloudKeyPoses6D != nullptr) {  // 下周看看为啥为空
        //   std::cout << "~~~~~~~~~开始执行合并地图~~~~~~~~~" << std::endl;
        //   auto globalmap_ = globalmap_stable;
        //   auto tempCloudKeyPoses_ = action_client->temporaryCloudKeyPoses6D;
        //   // std::async(std::launch::async, [this, globalmap_, tempCloudKeyPoses_] { this->mergeMap(globalmap_stable, action_client->temporaryCloudKeyPoses6D); });
        //   // std::async(std::launch::async, &HdlLocalizationNodelet::mergeMap, this, globalmap_stable, action_client->temporaryCloudKeyPoses6D);错误
        //   mergeMap(globalmap_stable, action_client->temporaryCloudKeyPoses6D);  // result->temporary_odometry
        //   std::cout << "~~~~~~~~~绕过合并地图~~~~~~~~~" << std::endl;
        // } else {
        //   std::cout << "~~~~~~~~~temporaryCloudKeyPoses6D is null~~~~~~~~~" << std::endl;
        // }
        // goodToMergeMap = false;
        // action_client.reset();
      }
    }
  }

  /**
   * @brief callback for point cloud data
   * @param points_msg
   */
  void points_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>());
    if (!globalmap) {
      NODELET_ERROR("globalmap has not been received!!");
      return;
    }
    std::cout << "~~~~~~~~~points_callback: received_points_flag~~~~~~~~~" << std::endl;
    received_points_flag = true;
    last_received_time = ros::WallTime::now();

    const auto& stamp = points_msg->header.stamp;
    pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *pcl_cloud);

    if (pcl_cloud->empty()) {
      NODELET_ERROR("cloud is empty!!");
      return;
    }

    // transform pointcloud into odom_child_frame_id
    std::string tfError;
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if (this->tf_buffer.canTransform(odom_child_frame_id, pcl_cloud->header.frame_id, stamp, ros::Duration(0.1), &tfError)) {  //! ros::Duration(0.1)
      if (!pcl_ros::transformPointCloud(odom_child_frame_id, *pcl_cloud, *cloud, this->tf_buffer)) {
        NODELET_ERROR("point cloud cannot be transformed into target frame!!");
        return;
      }
    } else {
      NODELET_ERROR(tfError.c_str());
      return;
    }

    auto filtered = downsample(cloud);
    last_scan = filtered;

    if (relocalizing) {
      delta_estimater->add_frame(filtered);
    }

    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
    if (!pose_estimator) {
      NODELET_ERROR("waiting for initial pose input!!");
      return;
    }
    Eigen::Matrix4f before = pose_estimator->matrix();

    // predict
    if (!use_imu) {
      pose_estimator->predict(stamp);
    } else {
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      auto imu_iter = imu_data.begin();
      for (imu_iter; imu_iter != imu_data.end(); imu_iter++) {
        if (stamp < (*imu_iter)->header.stamp) {
          break;
        }
        const auto& acc = (*imu_iter)->linear_acceleration;
        const auto& gyro = (*imu_iter)->angular_velocity;
        double acc_sign = invert_acc ? -1.0 : 1.0;
        double gyro_sign = invert_gyro ? -1.0 : 1.0;
        pose_estimator->predict((*imu_iter)->header.stamp, acc_sign * Eigen::Vector3f(acc.x, acc.y, acc.z), gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
      }
      imu_data.erase(imu_data.begin(), imu_iter);
    }

    // odometry-based prediction
    ros::Time last_correction_time = pose_estimator->last_correction_time();
    if (private_nh.param<bool>("enable_robot_odometry_prediction", false) && !last_correction_time.isZero()) {
      geometry_msgs::TransformStamped odom_delta;
      if (tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, ros::Duration(0.1))) {
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, ros::Duration(0));
      } else if (tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, ros::Time(0), robot_odom_frame_id, ros::Duration(0))) {
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, ros::Time(0), robot_odom_frame_id, ros::Duration(0));
      }

      if (odom_delta.header.stamp.isZero()) {
        NODELET_WARN_STREAM("failed to look up transform between " << cloud->header.frame_id << " and " << robot_odom_frame_id);
      } else {
        Eigen::Isometry3d delta = tf2::transformToEigen(odom_delta);
        pose_estimator->predict_odom(delta.cast<float>().matrix());
      }
    }

    // correct
    if (!action_client) {
      aligned = pose_estimator->correct(stamp, filtered);

    } else if (action_client && TMM_cancel_flag == 9) {
      float x = action_client->feedback_odomAftMapped.pose.pose.position.x;
      float y = action_client->feedback_odomAftMapped.pose.pose.position.y;
      float z = action_client->feedback_odomAftMapped.pose.pose.position.z;

      // 提取方向（四元数）
      float qx = action_client->feedback_odomAftMapped.pose.pose.orientation.x;
      float qy = action_client->feedback_odomAftMapped.pose.pose.orientation.y;
      float qz = action_client->feedback_odomAftMapped.pose.pose.orientation.z;
      float qw = action_client->feedback_odomAftMapped.pose.pose.orientation.w;

      Eigen::Quaternionf quat(qw, qx, qy, qz);
      Eigen::Matrix3f rotation_matrix = quat.toRotationMatrix();
      Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
      init_guess.block<3, 3>(0, 0) = rotation_matrix;
      init_guess(0, 3) = x;
      init_guess(1, 3) = y;
      init_guess(2, 3) = z;

      pcl::fromROSMsg(action_client->frame_point_cloud, *aligned);
      registration->setInputSource(aligned);
      registration->align(*aligned, init_guess);

      Eigen::Matrix4f trans = registration->getFinalTransformation();
      Eigen::Vector3f p = trans.block<3, 1>(0, 3);
      Eigen::Quaternionf q(trans.block<3, 3>(0, 0));

      if (quat.coeffs().dot(q.coeffs()) < 0.0f) {
        q.coeffs() *= -1.0f;
      }

      NODELET_INFO("endPriorPose caculated at point_callback()!!");
      std::lock_guard<std::mutex> lock(pose_estimator_mutex);
      // const auto& p = pose_msg->pose.pose.position;
      // const auto& q = pose_msg->pose.pose.orientation;
      pose_estimator.reset(new hdl_localization::PoseEstimator(
        registration,
        Eigen::Vector3f(p.x(), p.y(), p.z()),
        Eigen::Quaternionf(q.w(), q.x(), q.y(), q.z()),
        private_nh.param<double>("cool_time_duration", 0.5)));
    }

    latest_pose = eigenMatrixToPoseStamped(pose_estimator->matrix());  //! 传递最新的pose给fastlio  实际是上一帧的位姿，传给fastlio后并不是准确的第一帧位姿

    if (aligned_pub.getNumSubscribers()) {
      aligned->header.frame_id = "map";
      aligned->header.stamp = cloud->header.stamp;
      if (action_client) {
        if (!action_client->frame_point_cloud.data.empty()) {
          aligned_pub.publish(action_client->frame_point_cloud);
          pcl::fromROSMsg(action_client->frame_point_cloud, *aligned);  //! 用fastlio对齐的点云更新aligned   ---> Done
          std::cout << "~~~~~~~~~用fastlio对齐的点云更新aligned   ---> Done~~~~~~~~~" << std::endl;
        } else {
          // frame_point_cloud is empty
          ROS_INFO("frame_point_cloud is empty");
        }
      } else {
        aligned_pub.publish(aligned);
      }
    }

    if (status_pub.getNumSubscribers()) {
      if (action_client) {
        pcl::PointCloud<PointT>::Ptr output(new pcl::PointCloud<PointT>());
        // pcl::fromROSMsg(action_client->frame_point_cloud, *output);
        publish_scan_matching_status(points_msg->header, aligned);
      } else {
        publish_scan_matching_status(points_msg->header, aligned);
      }
    }

    publish_odometry(points_msg->header.stamp, pose_estimator->matrix());

    if (TMM_cancel_flag == 9) {                                                            //! 注意这个5，但是取消action的标志是6
      TMM_end_priorPose[count_needMergeMap + 1] = registration->getFinalTransformation();  // 满足条件后再取值
    }
    // TMM_end_priorPose[count_needMergeMap + 1] = pose_estimator->matrix();  // 包播放完毕后但是未满足TMM_cancel_flag == 9
  }

  geometry_msgs::PoseStamped eigenMatrixToPoseStamped(const Eigen::Matrix4f& pose_matrix) {
    geometry_msgs::PoseStamped pose_stamped;

    // 提取位置
    pose_stamped.pose.position.x = pose_matrix(0, 3);
    pose_stamped.pose.position.y = pose_matrix(1, 3);
    pose_stamped.pose.position.z = pose_matrix(2, 3);

    // 提取方向（旋转矩阵转换为四元数）
    Eigen::Matrix3f rotation_matrix = pose_matrix.block<3, 3>(0, 0);
    Eigen::Quaternionf quaternion(rotation_matrix);

    pose_stamped.pose.orientation.x = quaternion.x();
    pose_stamped.pose.orientation.y = quaternion.y();
    pose_stamped.pose.orientation.z = quaternion.z();
    pose_stamped.pose.orientation.w = quaternion.w();

    // 设置时间戳和坐标系ID（根据需要设置）
    pose_stamped.header.stamp = ros::Time::now();
    pose_stamped.header.frame_id = "map";

    return pose_stamped;
  }

  /**
   * @brief callback for globalmap input
   * @param points_msg
   */
  void globalmap_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    NODELET_INFO("globalmap received!");
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *cloud);
    globalmap = cloud;

    registration->setInputTarget(globalmap);

    if (use_global_localization) {
      NODELET_INFO("set globalmap for global localization!");
      hdl_global_localization::SetGlobalMap srv;
      pcl::toROSMsg(*globalmap, srv.request.global_map);

      if (!set_global_map_service.call(srv)) {
        NODELET_INFO("failed to set global map");
      } else {
        NODELET_INFO("done");
      }
    }
  }

  /**
   * @brief perform global localization to relocalize the sensor position
   * @param
   */
  bool relocalize(std_srvs::EmptyRequest& req, std_srvs::EmptyResponse& res) {
    if (last_scan == nullptr) {
      NODELET_INFO_STREAM("no scan has been received");
      return false;
    }

    relocalizing = true;
    delta_estimater->reset();
    pcl::PointCloud<PointT>::ConstPtr scan = last_scan;

    hdl_global_localization::QueryGlobalLocalization srv;
    pcl::toROSMsg(*scan, srv.request.cloud);
    srv.request.max_num_candidates = 1;

    if (!query_global_localization_service.call(srv) || srv.response.poses.empty()) {
      relocalizing = false;
      NODELET_INFO_STREAM("global localization failed");
      return false;
    }

    const auto& result = srv.response.poses[0];

    NODELET_INFO_STREAM("--- Global localization result ---");
    NODELET_INFO_STREAM("Trans :" << result.position.x << " " << result.position.y << " " << result.position.z);
    NODELET_INFO_STREAM("Quat  :" << result.orientation.x << " " << result.orientation.y << " " << result.orientation.z << " " << result.orientation.w);
    NODELET_INFO_STREAM("Error :" << srv.response.errors[0]);
    NODELET_INFO_STREAM("Inlier:" << srv.response.inlier_fractions[0]);

    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.linear() = Eigen::Quaternionf(result.orientation.w, result.orientation.x, result.orientation.y, result.orientation.z).toRotationMatrix();
    pose.translation() = Eigen::Vector3f(result.position.x, result.position.y, result.position.z);
    pose = pose * delta_estimater->estimated_delta();

    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    pose_estimator.reset(
      new hdl_localization::PoseEstimator(registration, pose.translation(), Eigen::Quaternionf(pose.linear()), private_nh.param<double>("cool_time_duration", 0.5)));

    relocalizing = false;

    return true;
  }

  /**
   * @brief callback for initial pose input ("2D Pose Estimate" on rviz)
   * @param pose_msg
   */
  void initialpose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg) {
    NODELET_INFO("initial pose received!!");
    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    const auto& p = pose_msg->pose.pose.position;
    const auto& q = pose_msg->pose.pose.orientation;
    pose_estimator.reset(new hdl_localization::PoseEstimator(
      registration,
      Eigen::Vector3f(p.x, p.y, p.z),
      Eigen::Quaternionf(q.w, q.x, q.y, q.z),
      private_nh.param<double>("cool_time_duration", 0.5)));
  }

  // void odometryCallback(const nav_msgs::Odometry::ConstPtr& msg) {  // 处理订阅的里程计消息
  //   // std::lock_guard<std::mutex> lock(pose_mutex);
  //   latest_pose.header = msg->header;
  //   latest_pose.pose = msg->pose.pose;
  // }

  // geometry_msgs::PoseStamped getLatestPose() {
  //   // std::lock_guard<std::mutex> lock(pose_mutex);
  //   return latest_pose;
  // }

  /**
   * @brief downsampling
   * @param cloud   input cloud
   * @return downsampled cloud
   */
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if (!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  /**
   * @brief publish odometry
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */
  void publish_odometry(const ros::Time& stamp, const Eigen::Matrix4f& pose) {
    // broadcast the transform over tf
    if (tf_buffer.canTransform(robot_odom_frame_id, odom_child_frame_id, ros::Time(0))) {
      geometry_msgs::TransformStamped map_wrt_frame = tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
      map_wrt_frame.header.stamp = stamp;
      map_wrt_frame.header.frame_id = odom_child_frame_id;
      map_wrt_frame.child_frame_id = "map";

      geometry_msgs::TransformStamped frame_wrt_odom = tf_buffer.lookupTransform(robot_odom_frame_id, odom_child_frame_id, ros::Time(0), ros::Duration(0.1));
      Eigen::Matrix4f frame2odom = tf2::transformToEigen(frame_wrt_odom).cast<float>().matrix();

      geometry_msgs::TransformStamped map_wrt_odom;
      tf2::doTransform(map_wrt_frame, map_wrt_odom, frame_wrt_odom);

      tf2::Transform odom_wrt_map;
      tf2::fromMsg(map_wrt_odom.transform, odom_wrt_map);
      odom_wrt_map = odom_wrt_map.inverse();

      geometry_msgs::TransformStamped odom_trans;
      odom_trans.transform = tf2::toMsg(odom_wrt_map);
      odom_trans.header.stamp = stamp;
      odom_trans.header.frame_id = "map";
      odom_trans.child_frame_id = robot_odom_frame_id;

      if (action_client) {
        tf_broadcaster.sendTransform(action_client->feedback_odom_trans);  //! 可能需要判断是不是时间上最新的数据
        odom_trans = action_client->feedback_odom_trans;                   // 需要更新使用feedback_odom_trans来更新odom_trans
      } else {
        tf_broadcaster.sendTransform(odom_trans);
      }
    } else {
      geometry_msgs::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
      odom_trans.header.stamp = stamp;
      odom_trans.header.frame_id = "map";
      odom_trans.child_frame_id = odom_child_frame_id;
      if (action_client) {
        tf_broadcaster.sendTransform(action_client->feedback_odom_trans);
        odom_trans = action_client->feedback_odom_trans;  //! 用fastlio的tf更新odom_trans   ---> Done
        std::cout << "~~~~~~~tf_broadcaster.sendTransform(action_client->feedback_odom_trans);~~~~~~~~~~" << std::endl;
      } else {
        tf_broadcaster.sendTransform(odom_trans);
      }
    }

    // publish the transform
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "map";
    odom.child_frame_id = "velodyne";

    tf::poseEigenToMsg(Eigen::Isometry3d(pose.cast<double>()), odom.pose.pose);
    odom.child_frame_id = odom_child_frame_id;
    odom.twist.twist.linear.x = 0.0;  // 线速度
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;  // 角速度

    if (action_client) {
      pose_pub.publish(action_client->feedback_odomAftMapped);
      odom = action_client->feedback_odomAftMapped;  //! 用fastlio的里程计更新odom   ---> Done
      std::cout << "~~~~~~~pose_pub.publish(action_client->feedback_odomAftMapped);~~~~~~~~~~" << std::endl;
    } else {
      pose_pub.publish(odom);
    }

    // 标记当前帧的位置方圆5米的全局地图点云
    // markPointsWithinRadius(globalmap_stable, odom.pose.pose, 5.0);//!不需要标记了，采用光路体素占据检查来去除全局地图点
  }

  /**
   * @brief publish scan matching status information
   */
  void publish_scan_matching_status(const std_msgs::Header& header, pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned) {
    ScanMatchingStatus status;
    status.header = header;

    status.has_converged = registration->hasConverged();
    status.matching_error = 0.0;

    const double max_correspondence_dist = private_nh.param<double>("status_max_correspondence_dist", 0.5);
    const double max_valid_point_dist = private_nh.param<double>("status_max_valid_point_dist", 25.0);

    int num_inliers = 0;
    int num_valid_points = 0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_dists;
    for (int i = 0; i < aligned->size(); i++) {
      const auto& pt = aligned->at(i);
      if (pt.getVector3fMap().norm() > max_valid_point_dist) {  // 最大距离来筛选有效点
        continue;
      }
      num_valid_points++;

      registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      if (k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
        status.matching_error += k_sq_dists[0];
        num_inliers++;
      }
    }

    status.matching_error /= num_inliers;
    status.inlier_fraction = static_cast<float>(num_inliers) / std::max(1, num_valid_points);
    hdl_inlier_ratio = status.inlier_fraction;  //! 类的成员变量传递内点状态
    std::cout << "~~~~~~~~~hdl_inlier_ratio~~~~~~~~~:" << hdl_inlier_ratio << std::endl;
    status.relative_pose = tf2::eigenToTransform(Eigen::Isometry3d(registration->getFinalTransformation().cast<double>())).transform;  // 这个没变，帧间变换矩阵

    status.prediction_labels.reserve(2);
    status.prediction_errors.reserve(2);

    std::vector<double> errors(6, 0.0);

    if (pose_estimator->wo_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = "without_pred";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->wo_prediction_error().get().cast<double>())).transform);
    }

    if (pose_estimator->imu_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = use_imu ? "imu" : "motion_model";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->imu_prediction_error().get().cast<double>())).transform);
    }

    if (pose_estimator->odom_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = "odom";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->odom_prediction_error().get().cast<double>())).transform);
    }

    status_pub.publish(status);
    // std::cout << "~~~~~~~status_pub.publish(status);~~~~~~~~~~" << std::endl;
  }

  void check_inlier_ratio(double hdl_inlier_ratio) {
    if (hdl_inlier_ratio < inlier_threshold_) {
      if (!action_client) {
        action_client = std::make_unique<TMMClient>(nh);
        ROS_INFO("Inlier ratio below threshold, sent new goal to action server with latest odometry data.");
        geometry_msgs::PoseStamped current_pose = latest_pose;
        // geometry_msgs::PoseStamped current_pose = getLatestPose();  // 这个位姿应该由hdl最新的位姿给
        // current_pose.header.stamp = ros::Time::now();
        // current_pose.header.frame_id = "camera_init";
        action_client->sendPose(current_pose, true);  // 这里的true就是触发TMM信号
        std::cout << "~~~~~~~~~已经发送当前位姿作为fastlio的全局位姿~~~~~~~~~" << std::endl;
        // 打印 position 信息
        ROS_INFO("Pose:");
        ROS_INFO("  Position:");
        ROS_INFO("    x: %f", current_pose.pose.position.x);
        ROS_INFO("    y: %f", current_pose.pose.position.y);
        ROS_INFO("    z: %f", current_pose.pose.position.z);

        // 打印 orientation 信息
        ROS_INFO("  Orientation:");
        ROS_INFO("    x: %f", current_pose.pose.orientation.x);
        ROS_INFO("    y: %f", current_pose.pose.orientation.y);
        ROS_INFO("    z: %f", current_pose.pose.orientation.z);
        ROS_INFO("    w: %f", current_pose.pose.orientation.w);
      }
      TMM_cancel_flag = 0;
    } else if (hdl_inlier_ratio < 0.95) {  //! 这里应该设计一个满足hdl_inlier_ratio >
      //! inlier_threshold_的记数count，当count大于一定值时，再将action_client重置，以避免在inlier_threshold_附近反复切换
      TMM_cancel_flag = 0;
      // if ((action_client && !action_client->isDone() && TMM_cancel_flag >= 6) || (!received_points_flag && action_client)) {
      //   std::cout << "~~~~~~~~~已经满足连续5次内点阈值条件,可以执行合并地图~~~~~~~~~" << std::endl;
      //   // Cancel the goal if action is still active
      //   action_client->cancelGoal();
      //   action_client.reset();  // 调用析构,释放内存,指针重置
      //   TMM_cancel_flag = 0;
      //   ROS_INFO("Inlier ratio above threshold 5 times, cancel the goal.");
      //   goodToMergeMap = true;
      // }
    } else if (hdl_inlier_ratio >= 0.985 && hdl_inlier_ratio <= 0.9999) {
      TMM_cancel_flag++;
    }
  }

  void status_callback(const ScanMatchingStatus::ConstPtr& msg) {
    // hdl_inlier_ratio = msg->inlier_fraction;
    // std::cout << "~~~~~~~~~hdl_inlier_ratio~~~~~~~~~" << hdl_inlier_ratio << std::endl;  //! 这个hdl_inlier_ratio也有可能是上一帧的？
    check_inlier_ratio(hdl_inlier_ratio);

    // 检查是否长时间未接收到消息
    // if (received_points_flag && (ros::Time::now() - last_received_time).toSec() > 5.0) {
    //   ROS_INFO("Bag playback finished or no new messages received");
    //   received_points_flag = false;
    // }

    // // 如果可以合并地图
    // if (goodToMergeMap) {  // 可以合并地图或者播包结束，则合并地图
    //   if (action_client->temporaryCloudKeyPoses6D != nullptr) {
    //     std::cout << "~~~~~~~~~开始执行合并地图~~~~~~~~~" << std::endl;
    //     auto globalmap_ = globalmap_stable;
    //     auto tempCloudKeyPoses_ = action_client->temporaryCloudKeyPoses6D;
    //     std::async(std::launch::async, [this, globalmap_, tempCloudKeyPoses_] { this->mergeMap(globalmap_stable, action_client->temporaryCloudKeyPoses6D); });
    //     // std::async(std::launch::async, &HdlLocalizationNodelet::mergeMap, this, globalmap_stable, action_client->temporaryCloudKeyPoses6D);
    //     // mergeMap(globalmap_stable, action_client->temporaryCloudKeyPoses6D);  // result->temporary_odometry
    //   } else {
    //     ROS_WARN("temporaryCloudKeyPoses6D is null");
    //   }
    //   goodToMergeMap = false;
    // }
  }

  void markPointsWithinRadius(pcl::PointCloud<PointT>::Ptr& globalmap_stable, const geometry_msgs::Pose& pose, float radius) {
    // 构建KD树
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(globalmap_stable);
    // 设置搜索点
    PointT searchPoint;
    searchPoint.x = pose.position.x;
    searchPoint.y = pose.position.y;
    searchPoint.z = pose.position.z;

    // 半径搜索
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
      for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i) {
        globalmap_stable->points[pointIdxRadiusSearch[i]].intensity = std::numeric_limits<float>::quiet_NaN();  // 将radius半径内的点的intensity标记为NaN
      }
    }
  }  //! 未被使用，被光路占据检测替代

  void mergeMap(pcl::PointCloud<PointT>::Ptr& globalmap, pcl::PointCloud<PointTypePose>::Ptr& temporaryCloudKeyPoses6D, int count_needMergeMap) {
    std::cout << " DO gtsam optimization here" << std::endl;

    // 初始化gtsam
    int priorNode = 0;
    gtsam::NonlinearFactorGraph gtSAMgraphTM;
    gtsam::Values initialEstimateTM;
    gtsam::ISAM2* isamTM;
    gtsam::Values isamCurrentEstimateTM;
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    isamTM = new gtsam::ISAM2(parameters);  // 配置 ISAM2 的参数 parameters 并创建 ISAM2 实例 isamTM
    std::cout << 1 << std::endl;

    // 添加先验因子  // rad*rad, meter*meter
    gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1).finished());
    std::cout << 2 << std::endl;

    gtsam::Pose3 posePrior = pclPointTogtsamPose3(temporaryCloudKeyPoses6D->points[priorNode]);
    // 获取平移部分
    std::cout << "TMM_end_priorPose " << TMM_end_priorPose.size() << std::endl;
    gtsam::Point3 translation_posePrior = posePrior.translation();
    std::cout << "translation_posePrior: " << translation_posePrior.x() << ", " << translation_posePrior.y() << ", " << translation_posePrior.z() << std::endl;

    // 获取旋转部分
    gtsam::Rot3 rotation_posePrior = posePrior.rotation();
    gtsam::Quaternion quaternion_posePrior = rotation_posePrior.toQuaternion();
    std::cout << "Rotation (quaternion_posePrior): " << quaternion_posePrior.w() << ", " << quaternion_posePrior.x() << ", " << quaternion_posePrior.y() << ", "
              << quaternion_posePrior.z() << std::endl;

    // 获取旋转的Roll, Pitch, Yaw
    gtsam::Vector3 rpy_posePrior = rotation_posePrior.rpy();
    std::cout << "Rotation (rpy_posePrior:roll, pitch, yaw): " << rpy_posePrior(0) << ", " << rpy_posePrior(1) << ", " << rpy_posePrior(2) << std::endl;

    gtSAMgraphTM.add(gtsam::PriorFactor<gtsam::Pose3>(priorNode, posePrior, priorNoise));  // 获取先验位姿 posePrior，并将其添加到因子图中。
    initialEstimateTM.insert(priorNode, posePrior);                                        // 将先验位姿插入初始估计中
    std::cout << 3 << std::endl;

    // 检查临时关键帧点云的大小
    int tempSize = temporaryCloudKeyPoses6D->points.size();
    if (tempSize < 3) return;
    // 循环添加里程计因子
    for (int i = priorNode; i < tempSize - 2; i++) {
      gtsam::noiseModel::Diagonal::shared_ptr odometryNoise =
        gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3).finished());  // 定义里程计噪声模型 odometryNoise。
      gtsam::Pose3 poseFrom = pclPointTogtsamPose3(temporaryCloudKeyPoses6D->points[i]);  // 获取当前和下一个关键帧的位姿 poseFrom 和 poseTo。
      gtsam::Pose3 poseTo = pclPointTogtsamPose3(temporaryCloudKeyPoses6D->points[i + 1]);
      gtSAMgraphTM.add(gtsam::BetweenFactor<gtsam::Pose3>(i, i + 1, poseFrom.between(poseTo), odometryNoise));  // 添加里程计因子到因子图中。
      initialEstimateTM.insert(i + 1, poseTo);
      // std::cout << 4 << std::endl;

      // update iSAM更新 ISAM2 并清空因子图和初始估计。TMM_end_priorPose
      isamTM->update(gtSAMgraphTM, initialEstimateTM);
      isamTM->update();
      gtSAMgraphTM.resize(0);
      initialEstimateTM.clear();
      // std::cout << 5 << std::endl;
    }
    gtsam::Pose3 poseCorr = Matrix4f2gtsamPose(TMM_end_priorPose[count_needMergeMap]);  // poseCorr是连续5次符合内点比例的最后一次（第4次）hdl位姿。
    // gtsam::Pose3 poseCorr = gtsam::Pose3(gtsam::Rot3::RzRyRx(-0.8776417, 0.5622363, 134.7281163), gtsam::Point3(4.8, 8.52503, 2.93349));

    // 获取平移部分
    std::cout << "TMM_end_priorPose " << TMM_end_priorPose.size() << std::endl;
    gtsam::Point3 translation = poseCorr.translation();
    std::cout << "Translation_poseCorr: " << translation.x() << ", " << translation.y() << ", " << translation.z() << std::endl;

    // 获取旋转部分
    gtsam::Rot3 rotation = poseCorr.rotation();
    gtsam::Quaternion quaternion = rotation.toQuaternion();
    std::cout << "Rotation_poseCorr (quaternion): " << quaternion.w() << ", " << quaternion.x() << ", " << quaternion.y() << ", " << quaternion.z() << std::endl;

    // 获取旋转的Roll, Pitch, Yaw
    gtsam::Vector3 rpy = rotation.rpy();
    std::cout << "Rotation_poseCorr (roll, pitch, yaw): " << rpy(0) << ", " << rpy(1) << ", " << rpy(2) << std::endl;

    std::cout << 6 << std::endl;

    // 添加最后的里程计因子和先验因子
    gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    gtsam::Pose3 poseFrom = pclPointTogtsamPose3(temporaryCloudKeyPoses6D->points[tempSize - 2]);
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(temporaryCloudKeyPoses6D->points[tempSize - 1]);
    gtSAMgraphTM.add(gtsam::BetweenFactor<gtsam::Pose3>(tempSize - 2, tempSize - 1, poseFrom.between(poseTo), odometryNoise));
    std::cout << 7 << std::endl;

    gtsam::noiseModel::Diagonal::shared_ptr corrNoise =
      gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3).finished());  // rad*rad, meter*meter
    gtSAMgraphTM.add(gtsam::PriorFactor<gtsam::Pose3>(tempSize - 1, poseCorr, corrNoise));
    initialEstimateTM.insert(tempSize - 1, poseCorr);
    std::cout << 8 << std::endl;

    // cout<<"before opt. "<< poseCorr.translation().x()<<" "<< poseCorr.translation().y()<<" "<< poseCorr.translation().z()<<endl;

    // update iSAM
    // 执行多次 ISAM2 更新确保优化收敛。清空因子图和初始估计。
    isamTM->update(gtSAMgraphTM, initialEstimateTM);
    isamTM->update();
    isamTM->update();
    isamTM->update();
    gtSAMgraphTM.resize(0);
    initialEstimateTM.clear();
    std::cout << 9 << std::endl;

    isamCurrentEstimateTM = isamTM->calculateEstimate();  // 获取当前的优化估计 isamCurrentEstimateTM。
    std::cout << 10 << std::endl;
    // cout<<"pose correction: "<<isamCurrentEstimateTM.size()<<endl;
    // 更新
    for (int i = priorNode; i < tempSize; i++) {
      // 更新临时点云关键帧的位姿。
      temporaryCloudKeyPoses6D->points[i].x = isamCurrentEstimateTM.at<gtsam::Pose3>(i).translation().x();
      temporaryCloudKeyPoses6D->points[i].y = isamCurrentEstimateTM.at<gtsam::Pose3>(i).translation().y();
      temporaryCloudKeyPoses6D->points[i].z = isamCurrentEstimateTM.at<gtsam::Pose3>(i).translation().z();
      temporaryCloudKeyPoses6D->points[i].roll = isamCurrentEstimateTM.at<gtsam::Pose3>(i).rotation().roll();
      temporaryCloudKeyPoses6D->points[i].pitch = isamCurrentEstimateTM.at<gtsam::Pose3>(i).rotation().pitch();
      temporaryCloudKeyPoses6D->points[i].yaw = isamCurrentEstimateTM.at<gtsam::Pose3>(i).rotation().yaw();
    }
    std::cout << 11 << std::endl;

    // // 移除global_map_stable中的标记点
    // globalmap->points.erase(
    //   std::remove_if(globalmap->points.begin(), globalmap->points.end(), [](const PointT& point) { return std::isnan(point.intensity); }),
    //   globalmap->points.end());
    // std::cout << 12 << std::endl;

    // 合并并发布地图
    // pcl::PointCloud<PointT>::Ptr cloudLocal(new pcl::PointCloud<PointT>());
    // std::string filename_TMM = TMM_saveDir + std::to_string(count_needMergeMap);
    // for (int i = 0; i < (int)tempSize; i++) {
    //   pcl::PointCloud<pcl::PointXYZI>::Ptr each_TMMpc = readAndTransformPointCloud(filename_TMM, i);
    //   *cloudLocal += *transformPointCloud(each_TMMpc, &temporaryCloudKeyPoses6D->points[i]);
    // }
    // *globalmap_stable += *cloudLocal;
    pcl::PointCloud<PointT>::Ptr cloudLocal(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr cloudTMM(new pcl::PointCloud<PointT>());
    std::string filename_TMM = TMM_saveDir + "/TMM_saveDir_" + std::to_string(count_needMergeMap);

    float voxel_size = 0.5;
    int length = 200;                                         // 100m / 0.5m
    int width = 200;                                          // 100m / 0.5m
    int height = 100;                                         // 50m / 0.5m
    VoxelGrid grid_frame(voxel_size, length, width, height);  // 没有放点进去

    // 处理全局地图的体素管理
    PointCloudVoxelInfo map_voxel_info(0.0, 0.0, 0.0);
    std::unordered_map<int, std::vector<pcl::PointXYZI>> map_voxel;
    std::cout << "去除路径上的体素点云前globalmap_stable的点的个数: " << globalmap->points.size() << std::endl;
    for (const auto& point : *globalmap) {
      int voxel_index = grid_frame.getVoxelIndex(point.x, point.y, point.z);
      map_voxel_info.occupied_voxel_indices.insert(voxel_index);  //! 获得map占据体素的索引
      map_voxel[voxel_index].push_back(point);
    }
    std::cout << "map_voxel_info.occupied_voxel_indices: " << map_voxel_info.occupied_voxel_indices.size() << std::endl;

    for (int i = 0; i < tempSize; i++) {
      pcl::PointCloud<pcl::PointXYZI>::Ptr each_TMMpc = readAndTransformPointCloud(filename_TMM, i);
      cloudLocal = transformPointCloud(each_TMMpc, &temporaryCloudKeyPoses6D->points[i]);  // velodyne坐标系点云变换到map坐标系
      *cloudTMM += *cloudLocal;
      // 对当前帧点云进行体素化管理，需要获得占据体素的索引、占据体素的中心位置、当前帧velodyne位姿、光路占据的体素索引
      PointCloudVoxelInfo frame_voxel_info(
        temporaryCloudKeyPoses6D->points[i].x,
        temporaryCloudKeyPoses6D->points[i].y,
        temporaryCloudKeyPoses6D->points[i].z);  //! 每一帧的位置中心
      for (const auto& point : *cloudLocal) {
        int voxel_index = grid_frame.getVoxelIndex(point.x, point.y, point.z);
        frame_voxel_info.occupied_voxel_indices.insert(voxel_index);  //! 获得frame占据体素的索引
      }
      std::cout << "frame_voxel_info.occupied_voxel_indices: " << frame_voxel_info.occupied_voxel_indices.size() << std::endl;
      for (auto index : frame_voxel_info.occupied_voxel_indices) {
        frame_voxel_info.voxelCenters.push_back(grid_frame.getVoxelCenter(index));  //! 获得frame占据体素的体素坐标
      }

      // 当前位姿处于全局体素的体素坐标xyz索引frame_voxel_info
      int ix = static_cast<int>(std::floor((frame_voxel_info.global_pose.x + grid_frame.origin.x()) / voxel_size));
      int iy = static_cast<int>(std::floor((frame_voxel_info.global_pose.y + grid_frame.origin.y()) / voxel_size));
      int iz = static_cast<int>(std::floor((frame_voxel_info.global_pose.z + grid_frame.origin.z()) / voxel_size));
      std::cout << "ix iy iz " << ix << " " << iy << " " << iz << std::endl;

      for (auto voxel_cord : frame_voxel_info.voxelCenters) {
        bresenham3D(ix, iy, iz, voxel_cord.x(), voxel_cord.y(), voxel_cord.z(), frame_voxel_info.unoccupied_voxels_indices, grid_frame);  //! frame光路占据的体素索引

        // unoccupied_voxels_indices记录了路径上未被占用的体素索引
      }
      // for (int i = 0; i < 1; i++) {
      //   bresenham3D(
      //     ix,
      //     iy,
      //     iz,
      //     frame_voxel_info.voxelCenters[i].x(),
      //     frame_voxel_info.voxelCenters[i].y(),
      //     frame_voxel_info.voxelCenters[i].z(),
      //     frame_voxel_info.unoccupied_voxels_indices,
      //     grid_frame);
      //   std::cout << "ix iy iz: " << ix << " " << iy << " " << iz << " " << std::endl;
      //   std::cout << "voxelCenters[i].xyz(): " << frame_voxel_info.voxelCenters[i].x() << " " << frame_voxel_info.voxelCenters[i].y() << " " <<
      //   frame_voxel_info.voxelCenters[i].z()
      //             << " " << std::endl;
      //   std::cout << "unoccupied_voxels_indices: " << frame_voxel_info.unoccupied_voxels_indices.size() << std::endl;
      //   for (auto it = frame_voxel_info.unoccupied_voxels_indices.begin(); it != frame_voxel_info.unoccupied_voxels_indices.end(); ++it) {
      //     int voxel_index = *it;
      //     std::cout << "unoccupied_voxels_indices的路径体素坐标:" << voxel_index << " " << grid_frame.getVoxelCenter(voxel_index).x() << " "
      //               << grid_frame.getVoxelCenter(voxel_index).y() << " " << grid_frame.getVoxelCenter(voxel_index).z() << std::endl;
      //   }
      // }

      // 上面处理好了一帧的体素管理,下面处理map和frame的占据对比
      for (auto unocc_index : frame_voxel_info.unoccupied_voxels_indices) {
        // if (frame_voxel_info.occupied_voxel_indices.find(unocc_index) == frame_voxel_info.occupied_voxel_indices.end()) {  // 如果路径体素未被当前帧占据

        if (map_voxel_info.occupied_voxel_indices.find(unocc_index) != map_voxel_info.occupied_voxel_indices.end()) {
          // std::cout << "unocc_index3: " << unocc_index << std::endl;
          // 获得unocc_index的全局地图系的xyx边界 Xmin、Xmax、
          // int ix = unocc_index / (grid_frame.grid_width * grid_frame.grid_height);
          // int iy = (unocc_index % (grid_frame.grid_width * grid_frame.grid_height)) / grid_frame.grid_height;
          // int iz = unocc_index % grid_frame.grid_height;  // 体素坐标系的xyz - 体素坐标的map原点
          // float Xmin = (ix * voxel_size) - grid_frame.origin.x();
          // float Xmax = (ix * voxel_size) - grid_frame.origin.x() + voxel_size;
          // float Ymin = (iy * voxel_size) - grid_frame.origin.y();
          // float Ymax = (iy * voxel_size) - grid_frame.origin.y() + voxel_size;
          // float Zmin = (iz * voxel_size) - grid_frame.origin.z();
          // float Zmax = (iz * voxel_size) - grid_frame.origin.z() + voxel_size;
          // auto it = globalmap_stable->points.begin();  //! 这里需要想办法剪枝 弃用
          // while (it != globalmap_stable->points.end()) {
          //   if (it->x >= Xmin && it->x < Xmax && it->y >= Ymin && it->y < Ymax && it->z >= Zmin && it->z < Zmax) {
          //     it = globalmap_stable->points.erase(it);
          //   } else {
          //     ++it;
          //   }
          // }

          auto it = map_voxel.find(unocc_index);
          if (it != map_voxel.end()) {  // 但是被全局地图占据
            // std::cout << "unocc_index4: " << unocc_index << std::endl;
            // for (auto pt_it = it->second.begin(); pt_it != it->second.end();) {
            //   // if (pt_it->x >= Xmin && pt_it->x < Xmax && pt_it->y >= Ymin && pt_it->y < Ymax && pt_it->z >= Zmin && pt_it->z < Zmax) {
            //   //   pt_it = it->second.erase(pt_it);
            //   // } else {
            //   //   ++pt_it;
            //   // }
            //   pt_it = it->second.erase(pt_it);
            // }
            // if (it->second.empty()) {
            //   map_voxel.erase(it);
            // }
            map_voxel.erase(it);
          }
        }
        // }
        //
      }
    }
    globalmap->points.clear();
    for (const auto& voxel : map_voxel) {
      globalmap->points.insert(globalmap->points.end(), voxel.second.begin(), voxel.second.end());
    }  // 去除全局地图上 每一帧的光路路径 被被占据的体素和点
    std::cout << "去除路径上的体素点云后globalmap_stable的点的个数: " << globalmap->points.size() << std::endl;
    *globalmap += *cloudTMM;  // 合并全局地图与TMM地图

    std::cout << 13 << std::endl;
    //! 这里还可以将globalmap_stable保存到本地文件，想想看怎么与原先每秒发布的全局地图怎么替换发布

    std::string filename_Merge_result = filename_Merge_result_ + "/fastlio_map_" + std::to_string(count_needMergeMap) + ".pcd";
    if (pcl::io::savePCDFile(filename_Merge_result, *globalmap) == -1) {
      ROS_ERROR("Couldn't save file %s", filename_Merge_result.c_str());
    }
    ROS_INFO("Saved global map to %s", filename_Merge_result.c_str());
    std::cout << 14 << std::endl;

    // // 将 globalmap_stable 的地址发布出去
    // std_msgs::String msg;
    // msg.data = filename;
    // map_update_pub.publish(msg);
    // ROS_INFO_STREAM("Published map update, map path: " << filename);
    // std::cout << 15 << std::endl;
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr readAndTransformPointCloud(std::string filename, int i) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    std::string filename_temp = filename + "/TMM_" + std::to_string(i) + ".pcd";
    // if (pcl::io::loadPCDFile(filename_temp, *cloud) == -1) {
    //   PCL_ERROR("Couldn't read file %s\n", filename_temp.c_str());
    //   return cloud;
    // }
    pcl::io::loadPCDFile(filename_temp, *cloud);
    return cloud;
  }

  gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint) {
    return gtsam::Pose3(
      gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
      gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
  }

  gtsam::Pose3 Matrix4f2gtsamPose(Eigen::Matrix4f& aff) {
    Eigen::Matrix3f rotation_matrix = aff.block<3, 3>(0, 0);
    Eigen::Vector3f translation_vector = aff.block<3, 1>(0, 3);

    Eigen::Matrix3d rotation_matrix_d = rotation_matrix.cast<double>();
    Eigen::Vector3d translation_vector_d = translation_vector.cast<double>();

    gtsam::Rot3 rotation_gtsam = gtsam::Rot3(rotation_matrix_d);
    gtsam::Point3 translation_gtsam(translation_vector_d.x(), translation_vector_d.y(), translation_vector_d.z());

    gtsam::Pose3 pose_gtsam(rotation_gtsam, translation_gtsam);

    return pose_gtsam;
  }

  pcl::PointCloud<PointT>::Ptr transformPointCloud(pcl::PointCloud<PointT>::Ptr cloudIn, PointTypePose* transformIn) {
    pcl::PointCloud<PointT>::Ptr cloudOut(new pcl::PointCloud<PointT>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

    // #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i) {
      const auto& pointFrom = cloudIn->points[i];
      cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
      cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
      cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
      cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
  }

  void bresenham3D(int x1, int y1, int z1, int x2, int y2, int z2, std::unordered_set<int>& unoccupied_voxels_indices, const VoxelGrid& grid) {
    int dx = abs(x2 - x1), dy = abs(y2 - y1), dz = abs(z2 - z1);
    int xs = (x2 > x1) ? 1 : -1;
    int ys = (y2 > y1) ? 1 : -1;
    int zs = (z2 > z1) ? 1 : -1;

    // Driving axis is X-axis
    if (dx >= dy && dx >= dz) {
      int p1 = 2 * dy - dx;
      int p2 = 2 * dz - dx;
      while (x1 != x2) {
        x1 += xs;
        if (p1 >= 0) {
          y1 += ys;
          p1 -= 2 * dx;
        }
        if (p2 >= 0) {
          z1 += zs;
          p2 -= 2 * dx;
        }
        p1 += 2 * dy;
        p2 += 2 * dz;
        // unoccupied_voxels_indices.insert(grid.getVoxelIndex(x1, y1, z1));  //! x1, y1, z1已经是体素坐标了！！！  getVoxelIndex()函数是对map坐标系下的点进行体素坐标的转化
        // std::cout << x1 << " " << y1 << " " << z1 << " " << std::endl;
        unoccupied_voxels_indices.insert(x1 * grid.grid_width * grid.grid_height + y1 * grid.grid_height + z1);
      }
    }
    // Driving axis is Y-axis
    else if (dy >= dx && dy >= dz) {
      int p1 = 2 * dx - dy;
      int p2 = 2 * dz - dy;
      while (y1 != y2) {
        y1 += ys;
        if (p1 >= 0) {
          x1 += xs;
          p1 -= 2 * dy;
        }
        if (p2 >= 0) {
          z1 += zs;
          p2 -= 2 * dy;
        }
        p1 += 2 * dx;
        p2 += 2 * dz;
        // unoccupied_voxels_indices.insert(grid.getVoxelIndex(x1, y1, z1));
        unoccupied_voxels_indices.insert(x1 * grid.grid_width * grid.grid_height + y1 * grid.grid_height + z1);
        // std::cout << x1 << " " << y1 << " " << z1 << " " << std::endl;
      }
    }
    // Driving axis is Z-axis
    else {
      int p1 = 2 * dy - dz;
      int p2 = 2 * dx - dz;
      while (z1 != z2) {
        z1 += zs;
        if (p1 >= 0) {
          y1 += ys;
          p1 -= 2 * dz;
        }
        if (p2 >= 0) {
          x1 += xs;
          p2 -= 2 * dz;
        }
        p1 += 2 * dy;
        p2 += 2 * dx;
        // unoccupied_voxels_indices.insert(grid.getVoxelIndex(x1, y1, z1));
        unoccupied_voxels_indices.insert(x1 * grid.grid_width * grid.grid_height + y1 * grid.grid_height + z1);
        // std::cout << x1 << " " << y1 << " " << z1 << " " << std::endl;
      }
    }
  }
  // 需要定义一个将feed_back的result的位姿转换为temporaryCloudKeyPoses6D的函数   可以在doneCb回调函数实现!!
  void saveMatricesToFile(const std::vector<Eigen::Matrix4f>& matrices, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::trunc);

    if (!outFile.is_open()) {
      std::cerr << "Unable to open file: " << filename << std::endl;
      return;
    }

    for (const auto& matrix : matrices) {
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          outFile << matrix(i, j);
          if (i != 3 || j != 3) {
            outFile << " ";
          }
        }
      }
      outFile << "\n";
    }
    outFile.close();
  }

private:
  // ROS
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;

  std::unique_ptr<TMMClient> action_client;
  // ros::Timer timer_; WallTimer
  ros::WallTimer timer_;
  double timeout_duration_;           // 5秒
  bool received_points_flag = false;  // 定义一个判断bag是否播报完的标志位
  ros::WallTime last_received_time;
  bool merge_maps_ = false;
  int count_needMergeMap = 0;
  int nums_needMergeMap = 0;
  std::string TMM_saveDir;
  std::string filename_Merge_result_;
  // std::string TMM_saveDir = "/home/alwen/project_learning/FAST-LIO_ws/src/FAST_LIO/PCD/TMM_saveDir_";

public:
  // parameter of voxel

  const float LIDAR_HEIGHT = 1.0;  // 小狗高度
  const int PC_NUM_RING = 20;
  const int PC_NUM_SECTOR = 60;
  const int PC_NUM_Z = 6;

  const double PC_MAX_RADIUS = 80.0;
  static constexpr double PC_MAX_Z = 6;

  std::string robot_odom_frame_id;
  std::string odom_child_frame_id;

  bool use_imu;
  bool invert_acc;
  bool invert_gyro;
  ros::Subscriber imu_sub;
  ros::Subscriber points_sub;
  ros::Subscriber globalmap_sub;
  ros::Subscriber initialpose_sub;

  // 存储最新的变换矩阵
  geometry_msgs::PoseStamped latest_pose;  // 发给fastlio的全局位姿作为其初值
  ros::Subscriber odometry_sub;
  std::mutex pose_mutex;

  ros::Publisher pose_pub;
  ros::Publisher aligned_pub;
  ros::Publisher status_pub;
  ros::Publisher map_update_pub;

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener;
  tf2_ros::TransformBroadcaster tf_broadcaster;

  // imu input buffer
  std::mutex imu_data_mutex;
  std::vector<sensor_msgs::ImuConstPtr> imu_data;

  // globalmap and registration method
  pcl::PointCloud<PointT>::Ptr globalmap;
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration;

  // 定义一个固定的全局地图
  pcl::PointCloud<PointT>::Ptr globalmap_stable;

  // pose estimator
  std::mutex pose_estimator_mutex;
  std::unique_ptr<hdl_localization::PoseEstimator> pose_estimator;

  // global localization
  bool use_global_localization;
  std::atomic_bool relocalizing;
  std::unique_ptr<DeltaEstimater> delta_estimater;

  pcl::PointCloud<PointT>::ConstPtr last_scan;
  ros::ServiceServer relocalize_server;
  ros::ServiceClient set_global_map_service;
  ros::ServiceClient query_global_localization_service;

  // TMM 参数
  float hdl_inlier_ratio;
  ros::Subscriber status_sub;
  double inlier_threshold_;
  int TMM_cancel_flag = 0;                         // 连续5次才能取消action请求
  std::vector<Eigen::Matrix4f> TMM_end_priorPose;  // 第五次时记录action去取消时hdl的定位结果，作为gtsam的位姿先验
  bool goodToMergeMap = false;
};
}  // namespace hdl_localization

PLUGINLIB_EXPORT_CLASS(hdl_localization::HdlLocalizationNodelet, nodelet::Nodelet)
