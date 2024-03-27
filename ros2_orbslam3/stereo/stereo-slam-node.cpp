#include "stereo-slam-node.hpp"

#include<opencv2/core/core.hpp>

using std::placeholders::_1;
using std::placeholders::_2;

StereoSlamNode::StereoSlamNode(ORB_SLAM3::System* pSLAM, const string &strSettingsFile, const string &strDoRectify)
:   Node("ORB_SLAM3_ROS2")
{
    SLAM = pSLAM;

    stringstream ss(strDoRectify);
    ss >> boolalpha >> doRectify;

    if (doRectify){

        cv::FileStorage fsSettings(strSettingsFile, cv::FileStorage::READ);
        if(!fsSettings.isOpened()){
            cerr << "ERROR: Wrong path to settings" << endl;
            assert(0);
        }

        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
        fsSettings["LEFT.K"] >> K_l;
        fsSettings["RIGHT.K"] >> K_r;

        fsSettings["LEFT.P"] >> P_l;
        fsSettings["RIGHT.P"] >> P_r;

        fsSettings["LEFT.R"] >> R_l;
        fsSettings["RIGHT.R"] >> R_r;

        fsSettings["LEFT.D"] >> D_l;
        fsSettings["RIGHT.D"] >> D_r;

        int rows_l = fsSettings["LEFT.height"];
        int cols_l = fsSettings["LEFT.width"];
        int rows_r = fsSettings["RIGHT.height"];
        int cols_r = fsSettings["RIGHT.width"];

        if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
                rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0){
            cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
            assert(0);
        }

        cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
        cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);
    }

    // left_sub = std::make_shared<message_filters::Subscriber<ImageMsg> >(shared_ptr<rclcpp::Node>(this), "camera/left");
    // right_sub = std::make_shared<message_filters::Subscriber<ImageMsg> >(shared_ptr<rclcpp::Node>(this), "camera/right");
    
    // syncApproximate = std::make_shared<message_filters::Synchronizer<approximate_sync_policy> >(approximate_sync_policy(10), *left_sub, *right_sub);
    // syncApproximate->registerCallback(&StereoSlamNode::GrabStereo, this);

    subImgLeft = this->create_subscription<ImageMsg>("cam0/image_quantized", 100, std::bind(&StereoSlamNode::GrabImageLeft, this, _1));
    subImgRight = this->create_subscription<ImageMsg>("cam1/image_quantized", 100, std::bind(&StereoSlamNode::GrabImageRight, this, _1));

    syncThread = new std::thread(&StereoSlamNode::SyncBothImages, this);
}   

StereoSlamNode::~StereoSlamNode()
{
    // Delete sync thread
    syncThread->join();
    delete syncThread;

    // Stop all threads
    SLAM->Shutdown();

    // Save camera trajectory
    SLAM->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}


void StereoSlamNode::GrabImageLeft(const ImageMsg::SharedPtr msgLeft)
{
    bufMutexLeft.lock();

    if (!imgLeftBuf.empty())
        imgLeftBuf.pop();
    imgLeftBuf.push(msgLeft);

    bufMutexLeft.unlock();
}

void StereoSlamNode::GrabImageRight(const ImageMsg::SharedPtr msgRight)
{
    bufMutexRight.lock();

    if (!imgRightBuf.empty())
        imgRightBuf.pop();
    imgRightBuf.push(msgRight);

    bufMutexRight.unlock();
}

cv::Mat StereoSlamNode::GetImage(const ImageMsg::SharedPtr msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;

    try
    {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }

    if (cv_ptr->image.type() == 0)
    {
        return cv_ptr->image.clone();
    }
    else
    {
        std::cerr << "Error image type" << std::endl;
        return cv_ptr->image.clone();
    }
}

void StereoSlamNode::SyncBothImages()
{
    const double maxTimeDiff = 0.01;

    while (1)
    {
        cv::Mat imLeft, imRight;
        double tImLeft = 0, tImRight = 0;
        if (!imgLeftBuf.empty() && !imgRightBuf.empty())
        {
            tImLeft = Utility::StampToSec(imgLeftBuf.front()->header.stamp);
            tImRight = Utility::StampToSec(imgRightBuf.front()->header.stamp);

            bufMutexRight.lock();
            while ((tImLeft - tImRight) > maxTimeDiff && imgRightBuf.size() > 1)
            {
                imgRightBuf.pop();
                tImRight = Utility::StampToSec(imgRightBuf.front()->header.stamp);
            }
            bufMutexRight.unlock();

            bufMutexLeft.lock();
            while ((tImRight - tImLeft) > maxTimeDiff && imgLeftBuf.size() > 1)
            {
                imgLeftBuf.pop();
                tImLeft = Utility::StampToSec(imgLeftBuf.front()->header.stamp);
            }
            bufMutexLeft.unlock();

            if ((tImLeft - tImRight) > maxTimeDiff || (tImRight - tImLeft) > maxTimeDiff)
            {
                std::cout << "big time difference" << std::endl;
                continue;
            }

            bufMutexLeft.lock();
            imLeft = GetImage(imgLeftBuf.front());
            imgLeftBuf.pop();
            bufMutexLeft.unlock();

            bufMutexRight.lock();
            imRight = GetImage(imgRightBuf.front());
            imgRightBuf.pop();
            bufMutexRight.unlock();

            if (bClahe)
            {
                clahe->apply(imLeft, imLeft);
                clahe->apply(imRight, imRight);
            }

            if (doRectify)
            {
                cv::remap(imLeft, imLeft, M1l, M2l, cv::INTER_LINEAR);
                cv::remap(imRight, imRight, M1r, M2r, cv::INTER_LINEAR);
            }

            SLAM->TrackStereo(imLeft, imRight, tImLeft);

            std::chrono::milliseconds tSleep(1);
            std::this_thread::sleep_for(tSleep);
        }
    }
}
