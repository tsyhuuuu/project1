#ifndef __STEREO_SLAM_NODE_HPP__
#define __STEREO_SLAM_NODE_HPP__

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include <cv_bridge/cv_bridge.h>

#include "System.h"
#include "Frame.h"
#include "Map.h"
#include "Tracking.h"

#include "utility.hpp"

using ImageMsg = sensor_msgs::msg::Image;

class StereoSlamNode : public rclcpp::Node
{
public:
    StereoSlamNode(ORB_SLAM3::System* pSLAM, const string &strSettingsFile, const string &strDoRectify);
    ~StereoSlamNode();

private:
    void GrabImageLeft(const ImageMsg::SharedPtr msgLeft);
    void GrabImageRight(const ImageMsg::SharedPtr msgRight);
    void SyncBothImages();
    cv::Mat GetImage(const ImageMsg::SharedPtr msg);

    rclcpp::Subscription<ImageMsg>::SharedPtr subImgLeft;
    rclcpp::Subscription<ImageMsg>::SharedPtr subImgRight;

    ORB_SLAM3::System *SLAM;
    std::thread *syncThread;

    // Image
    queue<ImageMsg::SharedPtr> imgLeftBuf, imgRightBuf;
    std::mutex bufMutexLeft, bufMutexRight;

    bool doRectify;
    bool doEqual;
    cv::Mat M1l, M2l, M1r, M2r;

    bool bClahe;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
};

#endif
