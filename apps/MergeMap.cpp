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

#include <iostream>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <eigen_conversions/eigen_msg.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>

#include <string>
#include <cmath>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include <boost/filesystem.hpp>

using PointT = pcl::PointXYZI;

struct PointXYZIRPYT {
  PCL_ADD_POINT4D;    // preferred way of adding a XYZ+padding
  PCL_ADD_INTENSITY;  // add intensity
  double roll;
  double pitch;
  double yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // ensure proper alignment
} EIGEN_ALIGN16;

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

// Register the point type with PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(
  PointXYZIRPYT,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(double, roll, roll)(double, pitch, pitch)(double, yaw, yaw)(double, time, time))
typedef PointXYZIRPYT PointTypePose;

std::vector<Eigen::Matrix4f> loadMatricesFromFile(const std::string& filename) {
  std::ifstream inFile(filename);
  std::vector<Eigen::Matrix4f> matrices;
  if (!inFile.is_open()) {
    std::cout << "Unable to open file: " << filename << std::endl;
    return matrices;
  }

  std::string line;
  while (std::getline(inFile, line)) {
    std::istringstream iss(line);
    Eigen::Matrix4f matrix;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        iss >> matrix(i, j);
      }
    }
    matrices.push_back(matrix);
  }

  inFile.close();
  return matrices;
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

int countSubdirectoriesWithPrefix(const boost::filesystem::path& dirPath, const std::string& prefix) {
  int count = 0;

  if (boost::filesystem::exists(dirPath) && boost::filesystem::is_directory(dirPath)) {
    for (const auto& entry : boost::filesystem::directory_iterator(dirPath)) {
      if (boost::filesystem::is_directory(entry.path()) && entry.path().filename().string().find(prefix) == 0) {
        count++;
      }
    }
  } else {
    std::cerr << "Directory does not exist or is not a directory: " << dirPath << std::endl;
  }

  return count;
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

void mergeMap(
  pcl::PointCloud<PointT>::Ptr& globalmap,
  pcl::PointCloud<PointTypePose>::Ptr& temporaryCloudKeyPoses6D,
  int count_needMergeMap,
  std::string TMM_each_path_,
  std::string output_map,
  std::vector<Eigen::Matrix4f> TMM_end_priorPose) {
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
  std::string filename_TMM = TMM_each_path_ + "/TMM_saveDir_" + std::to_string(count_needMergeMap);

  float voxel_size = 1.0;
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
    PointCloudVoxelInfo frame_voxel_info(temporaryCloudKeyPoses6D->points[i].x, temporaryCloudKeyPoses6D->points[i].y,
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
      if (frame_voxel_info.occupied_voxel_indices.find(unocc_index) == frame_voxel_info.occupied_voxel_indices.end()) {  // 如果路径体素未被当前帧占据

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

            map_voxel.erase(it);
          }
        }
      }
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

  std::string filename_Merge_result = output_map + "/Merge_map_0724" + std::to_string(count_needMergeMap) + ".pcd";
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

int main(int argc, char** argv) {
  ros::init(argc, argv, "merge_map_node");
  ros::NodeHandle nh;

  std::string prior_global_map, TMM_each_path_, output_map;

  bool goodToMergeMap;
  nh.getParam("prior_global_map", prior_global_map);
  nh.getParam("TMM_each_path_", TMM_each_path_);
  nh.getParam("output_map", output_map);
  nh.getParam("goodToMergeMap", goodToMergeMap);

  std::cout << "prior_global_map: " << prior_global_map << std::endl;
  std::cout << "TMM_each_path_: " << TMM_each_path_ << std::endl;
  std::cout << "output_map: " << output_map << std::endl;

  std::string prefix = "TMM_saveDir_";

  double downsample_resolution = nh.param<double>("downsample_resolution", 0.1);
  boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
  voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);

  // 定义一个全局固定的地图
  //  read globalmap from a pcd file
  pcl::PointCloud<PointT>::Ptr globalmap_stable(new pcl::PointCloud<PointT>());
  std::cout << "globalmap_stable: " << 1 << std::endl;
  pcl::io::loadPCDFile(prior_global_map, *globalmap_stable);
  std::cout << "prior_global_map: " << prior_global_map << std::endl;
  std::cout << "globalmap_stable: " << 2 << std::endl;

  voxelgrid->setInputCloud(globalmap_stable);
  voxelgrid->filter(*globalmap_stable);

  int count_needMergeMap = countSubdirectoriesWithPrefix(TMM_each_path_, prefix);
  std::cout << "count_needMergeMap: " << count_needMergeMap << std::endl;

  std::string TMM_end_priorPose_path = TMM_each_path_ + "/TMM_end_prior_pose.txt";
  std::vector<Eigen::Matrix4f> TMM_end_priorPose;
  TMM_end_priorPose = loadMatricesFromFile(TMM_end_priorPose_path);

  if (goodToMergeMap) {  // 可以合并地图或者播包结束，则合并地图。
    goodToMergeMap = false;
    for (int i = 1; i <= count_needMergeMap; i++) {
      // std::cout << "~~~~~~~~~ action_client->temporaryCloudKeyPoses6D~~~~~~~~~" << action_client->temporaryCloudKeyPoses6D->size() << std::endl;
      pcl::PointCloud<PointTypePose>::Ptr TMModom(new pcl::PointCloud<PointTypePose>());
      std::string TMM_each_path = TMM_each_path_ + "/TMM_saveDir_" + std::to_string(i);
      std::string TMM_each_path_odom = TMM_each_path + "/TMM_odom.txt";
      readOdometryFile(TMM_each_path_odom, TMModom);

      if (!TMModom->empty()) {
        std::cout << "~~~~~~~~~开始执行合并区域" << i << "的地图~~~~~~~~~" << TMModom->points.size() << std::endl;
        mergeMap(globalmap_stable, TMModom, i, TMM_each_path_, output_map, TMM_end_priorPose);
      } else {
        std::cout << "~~~~~~~~~temporaryCloudKeyPoses6D is null~~~~~~~~~" << std::endl;
      }
    }
  }
  ros::spin();
  return 0;
}
