
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the
// same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes,
                         std::vector<LidarPoint> &lidarPoints,
                         float shrinkFactor, cv::Mat &P_rect_xx,
                         cv::Mat &R_rect_xx, cv::Mat &RT) {
  // loop over all Lidar points and associate them to a 2D bounding box
  cv::Mat X(4, 1, cv::DataType<double>::type);
  cv::Mat Y(3, 1, cv::DataType<double>::type);

  for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1) {
    // assemble vector for matrix-vector-multiplication
    X.at<double>(0, 0) = it1->x;
    X.at<double>(1, 0) = it1->y;
    X.at<double>(2, 0) = it1->z;
    X.at<double>(3, 0) = 1;

    // project Lidar point into camera
    Y = P_rect_xx * R_rect_xx * RT * X;
    cv::Point pt;
    pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
    pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

    vector<vector<BoundingBox>::iterator>
        enclosingBoxes; // pointers to all bounding boxes which enclose the
                        // current Lidar point
    for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin();
         it2 != boundingBoxes.end(); ++it2) {
      // shrink current bounding box slightly to avoid having too many outlier
      // points around the edges
      cv::Rect smallerBox;
      smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
      smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
      smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
      smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

      // check wether point is within current bounding box
      if (smallerBox.contains(pt)) {
        enclosingBoxes.push_back(it2);
      }

    } // eof loop over all bounding boxes

    // check wether point has been enclosed by one or by multiple boxes
    if (enclosingBoxes.size() == 1) {
      // add Lidar point to bounding box
      enclosingBoxes[0]->lidarPoints.push_back(*it1);
    }

  } // eof loop over all Lidar points
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize,
                   cv::Size imageSize, bool bWait) {
  // create topview image
  cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

  for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1) {
    // create randomized color for current 3D object
    cv::RNG rng(it1->boxID);
    cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150),
                                      rng.uniform(0, 150));

    // plot Lidar points into top view image
    int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
    float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
    for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end();
         ++it2) {
      // world coordinates
      float xw =
          (*it2).x; // world position in m with x facing forward from sensor
      float yw = (*it2).y; // world position in m with y facing left from sensor
      xwmin = xwmin < xw ? xwmin : xw;
      ywmin = ywmin < yw ? ywmin : yw;
      ywmax = ywmax > yw ? ywmax : yw;

      // top-view coordinates
      int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
      int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

      // find enclosing rectangle
      top = top < y ? top : y;
      left = left < x ? left : x;
      bottom = bottom > y ? bottom : y;
      right = right > x ? right : x;

      // draw individual point
      cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
    }

    // draw enclosing rectangle
    cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),
                  cv::Scalar(0, 0, 0), 2);

    // augment object with some key data
    char str1[200], str2[200];
    sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
    putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50),
            cv::FONT_ITALIC, 2, currColor);
    sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
    putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125),
            cv::FONT_ITALIC, 2, currColor);
  }

  // plot distance markers
  float lineSpacing = 2.0; // gap between distance markers
  int nMarkers = floor(worldSize.height / lineSpacing);
  for (size_t i = 0; i < nMarkers; ++i) {
    int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) +
            imageSize.height;
    cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y),
             cv::Scalar(255, 0, 0));
  }

  // display image
  string windowName = "3D Objects";
  cv::namedWindow(windowName, 1);
  cv::imshow(windowName, topviewImg);

  if (bWait) {
    cv::waitKey(0); // wait for key to be pressed
  }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox,
                              std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr,
                              std::vector<cv::DMatch> &kptMatches) {
  std::vector<cv::DMatch> kpt_matches;
  for (auto iter = kptMatches.begin(); iter != kptMatches.end(); ++iter) {
    const auto &kpCurr = kptsCurr.at(iter->trainIdx);
    if (boundingBox.roi.contains(kpCurr.pt)) {
      kpt_matches.emplace_back(*iter);
    }
  }
  std::cout << "kpt_matches=" << kpt_matches.size() << std::endl;

  std::vector<double> dist_kpt_matches;
  for (auto iter = kpt_matches.begin(); iter != kpt_matches.end(); ++iter) {
    const auto &kp_curr = kptsCurr.at(iter->trainIdx);
    const auto &kp_prev = kptsPrev.at(iter->queryIdx);
    dist_kpt_matches.emplace_back(cv::norm(kp_curr.pt - kp_prev.pt));
  }
  const double mean =
      std::accumulate(dist_kpt_matches.begin(), dist_kpt_matches.end(), 0.0) /
      std::max(dist_kpt_matches.size(),
               static_cast<std::vector<double>::size_type>(1));
  const double var =
      std::accumulate(dist_kpt_matches.begin(), dist_kpt_matches.end(), 0.0,
                      [&mean](const double &sum, const double &x) {
                        const double d = x - mean;
                        return sum + d * d;
                      }) /
      std::max(dist_kpt_matches.size(),
               static_cast<std::vector<double>::size_type>(1));
  const double stdd = std::sqrt(var);

  // Keep all kpt matches dist < 2 * std
  for (auto iter = kpt_matches.begin(); iter != kpt_matches.end(); ++iter) {
    const auto &kp_curr = kptsCurr.at(iter->trainIdx);
    const auto &kp_prev = kptsPrev.at(iter->queryIdx);
    const double dist = cv::norm(kp_curr.pt - kp_prev.pt);
    if (dist < 2.0 * stdd) {
      boundingBox.kptMatches.emplace_back(*iter);
    }
  }
  //  std::cout << "kptMatches=" << boundingBox.kptMatches.size() << std::endl;
}

// Compute time-to-collision (TTC) based on keypoint correspondences in
// successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev,
                      std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate,
                      double &TTC, cv::Mat *visImg) {
  // compute distance ratios between all matched keypoints
  std::vector<double> dist_ratios; // stores the distance ratios for all
                                   // keypoints between curr. and prev. frame
  for (auto outer_it = kptMatches.begin(); outer_it != kptMatches.end() - 1;
       ++outer_it) { // outer kpt. loop

    // get current keypoint and its matched partner in the prev. frame
    cv::KeyPoint kp_outer_curr = kptsCurr.at(outer_it->trainIdx);
    cv::KeyPoint kp_outer_prev = kptsPrev.at(outer_it->queryIdx);

    for (auto inner_it = kptMatches.begin() + 1; inner_it != kptMatches.end();
         ++inner_it) { // inner kpt.-loop

      const double min_dist = 100.0; // min. required distance

      // get next keypoint and its matched partner in the prev. frame
      cv::KeyPoint kp_inner_curr = kptsCurr.at(inner_it->trainIdx);
      cv::KeyPoint kp_inner_prev = kptsPrev.at(inner_it->queryIdx);

      // compute distances and distance ratios
      double dist_curr = cv::norm(kp_outer_curr.pt - kp_inner_curr.pt);
      double dist_prev = cv::norm(kp_outer_prev.pt - kp_inner_prev.pt);

      if (dist_prev > std::numeric_limits<double>::epsilon() &&
          dist_curr >= min_dist) { // avoid division by zero

        const double dist_ratio = dist_curr / dist_prev;
        dist_ratios.push_back(dist_ratio);
      }
    } // eof inner loop over all matched kpts
  }   // eof outer loop over all matched kpts

  // only continue if list of distance ratios is not empty
  if (dist_ratios.size() == 0) {
    TTC = NAN;
    return;
  }

  std::sort(dist_ratios.begin(), dist_ratios.end());
  std::vector<double>::size_type mid_idx = dist_ratios.size() / 2;
  const double mid_dist_ratio =
      dist_ratios.size() % 2 == 0
          ? (dist_ratios[mid_idx - 1] + dist_ratios[mid_idx]) / 2.0
          : dist_ratios[mid_idx]; // compute median dist. ratio to remove
                                  // outlier influence

  const double dT = 1.0 / frameRate;
  TTC = -dT / (1 - mid_dist_ratio);
  std::cout << " CAMERA ttc=" << TTC << "s" << std::endl;
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate,
                     double &TTC) {
  // Assuming ego lane
  auto d_fun_median =
      [](const std::vector<LidarPoint> &lidar_pooints) -> double {
    std::vector<double> x;
    std::transform(lidar_pooints.begin(), lidar_pooints.end(),
                   std::back_inserter(x),
                   [](const LidarPoint &lp) -> double { return lp.x; });
    std::sort(x.begin(), x.end());
    std::vector<double>::size_type mid_idx = x.size() / 2;
    return x.size() % 2 == 0 ? (x[mid_idx - 1] + x[mid_idx]) / 2.0 : x[mid_idx];
  };
  const auto d1 = d_fun_median(lidarPointsCurr);
  const auto d0 = d_fun_median(lidarPointsPrev);
  const double dt = 1.0 / frameRate;
  const double den = std::abs(d0 - d1) > 0 ? std::abs(d0 - d1) : 1e-6;

  TTC = d1 * dt / den;
  std::cout << " LIDAR ttc=" << TTC << "s" << std::endl;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches,
                        std::map<int, int> &bbBestMatches, DataFrame &prevFrame,
                        DataFrame &currFrame) {
  std::map<int, std::vector<cv::DMatch>> query_bb_to_kpt_matches;
  std::map<int, std::vector<int>> train_idx_to_bb;

  for (auto match_iter = matches.begin(); match_iter != matches.end();
       ++match_iter) {
    // Query
    for (auto query_bb_iter = prevFrame.boundingBoxes.begin();
         query_bb_iter != prevFrame.boundingBoxes.end(); ++query_bb_iter) {
      const auto keypoint = prevFrame.keypoints.at(match_iter->queryIdx);
      if (query_bb_iter->roi.contains(keypoint.pt)) {
        //        query_idx_to_box_ids[match_iter->queryIdx].emplace_back(
        //            query_bb_iter->boxID);
        query_bb_to_kpt_matches[query_bb_iter->boxID].emplace_back(*match_iter);
      }
    }
    // Train
    for (auto train_bb_iter = currFrame.boundingBoxes.begin();
         train_bb_iter != currFrame.boundingBoxes.end(); ++train_bb_iter) {
      const auto keypoint = currFrame.keypoints.at(match_iter->trainIdx);
      if (train_bb_iter->roi.contains(keypoint.pt)) {
        //        train_box_id_to_kp_matches[train_bb_iter->boxID].emplace_back(
        //            *match_iter);
        train_idx_to_bb[match_iter->trainIdx].emplace_back(
            train_bb_iter->boxID);
      }
    }
  }

  // From query (prev) to train (curr) mapping.
  for (auto query_bb_iter = prevFrame.boundingBoxes.begin();
       query_bb_iter != prevFrame.boundingBoxes.end(); ++query_bb_iter) {
    std::map<int, int> counts;
    int max_curr_bb = 0, max_curr_count = 0;
    auto query_bb_to_kpt_matches_iter =
        query_bb_to_kpt_matches.find(query_bb_iter->boxID);
    if (query_bb_to_kpt_matches_iter != query_bb_to_kpt_matches.end()) {
      for (const auto &query_bb_kpt_matches :
           query_bb_to_kpt_matches_iter->second) {
        for (const auto &train_bb :
             train_idx_to_bb[query_bb_kpt_matches.trainIdx]) {
          ++counts[train_bb];
          if (counts[train_bb] >= max_curr_bb) {
            max_curr_count = counts[train_bb];
            max_curr_bb = train_bb;
          }
        }
      }
      //      std::cout << "prev_bb=" << query_bb_iter->boxID
      //                << " -> curr_bb=" << max_curr_bb << std::endl;
      bbBestMatches.emplace(query_bb_iter->boxID, max_curr_bb);
    }
  }
}

// Helpers
StatsFactory &StatsFactory::instance() {
  static StatsFactory instance;
  return instance;
}
void StatsFactory::add_record(const std::string &det, const std::string &des,
                              int img_index) {
  curr_idx = storage.size();
  storage.emplace_back(Stats(det, des, img_index));
  storage[curr_idx].id = curr_idx;
}
void StatsFactory::update_ttc_lidar(double ttc_lidar) {
  storage[curr_idx].ttc_lidar = ttc_lidar;
}
void StatsFactory::update_ttc_camera(double ttc_camera) {
  storage[curr_idx].ttc_camera = ttc_camera;
}
void StatsFactory::write(const std::string &output_path) {
  std::ofstream out;
  out.open(output_path);
  std::stringstream head_ss;
  head_ss << "det,des,id,img_idx,ttc_lidar,ttc_camera,ttc_diff";
  out << head_ss.str() << "\n";
  for (auto iter = storage.begin(); iter != storage.end(); ++iter) {
    std::stringstream ss;
    ss << iter->det << "," << iter->des << "," << iter->id << ","
       << iter->img_idx << "," << iter->ttc_lidar << "," << iter->ttc_camera
       << "," << (iter->ttc_lidar - iter->ttc_camera);
    out << ss.str() << "\n";
  }
  out.close();
}
