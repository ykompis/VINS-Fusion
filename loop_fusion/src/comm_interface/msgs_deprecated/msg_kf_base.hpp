#pragma once

#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>

// CoVINS
#include "../typedefs.hpp"

namespace comm_interface {

struct PreintegrationData{

    Eigen::Vector3d         acc;
    Eigen::Vector3d         gyr;
    Eigen::Vector3d         lin_bias_accel;
    Eigen::Vector3d         lin_bias_gyro;
    std::vector<double>     dt;
    std::vector<double>     lin_acc_x;
    std::vector<double>     lin_acc_y;
    std::vector<double>     lin_acc_z;
    std::vector<double>     ang_vel_x;
    std::vector<double>     ang_vel_y;
    std::vector<double>     ang_vel_z;

    template<class Archive> auto serialize( Archive & archive )->void {
        archive(acc,gyr,lin_bias_accel,lin_bias_gyro,
                dt,
                lin_acc_x,lin_acc_y,lin_acc_z,
                ang_vel_x,ang_vel_y,ang_vel_z);
    }
};

struct MsgKeyframeBase {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using idpair                        = TypeDefs::idpair;
    using TransformType                 = TypeDefs::TransformType;
    using Vector3Type                   = TypeDefs::Vector3Type;
    using KeypointVector                = TypeDefs::KeypointVector;
    using AorsVector                    = TypeDefs::AorsVector;

    using LandmarksMinimalType          = std::map<int, idpair>;
    using MsgTypeVector                 = std::vector<uint32_t>;

    struct compare_less{bool operator() (const MsgKeyframeBase &a, const MsgKeyframeBase &b) const;};

public:
    MsgKeyframeBase();
    MsgKeyframeBase(bool filesave);
    MsgKeyframeBase(MsgTypeVector msgtype);

    //Interfaces
    auto SetMsgType(int msg_size)                                       ->void;
    auto SetMsgType(MsgTypeVector msgtype)                              ->void;

    //Infrastructure
    MsgTypeVector           msg_type                                                            = std::vector<uint32_t>(5);     // size, is_update, ID of Keyframe, ???,;
    bool                    is_update_msg                                                       = false;
    bool                    save_to_file                                                        = false;                        // indicates that this KF will be saved to a file, not send over network

    //Identifier
    double                  timestamp;
    idpair                  id;

    //Calibration
    VICalibration           calibration;
    int                     img_dim_x_min;
    int                     img_dim_y_min;
    int                     img_dim_x_max;
    int                     img_dim_y_max;

    // KeyPoints and descriptors (all associated by an index)
    KeypointVector          keypoints_distorted;
    KeypointVector          keypoints_undistorted;
    AorsVector              keypoints_aors;                                                                                     //Angle,Octave,Response,Size
    cv::Mat                 descriptors;

    // SE3 Pose, Bias, Velocity
    TransformType           T_s_c                                                               = TransformType::Identity();    // Tranformation IMU-Cam

    TransformType           T_sref_s                                                            = TransformType::Identity();
    TransformType           T_w_s                                                               = TransformType::Identity();

    Vector3Type             velocity                                                            = Vector3Type::Zero();
    Vector3Type             bias_gyro                                                           = Vector3Type::Zero();
    Vector3Type             bias_accel                                                          = Vector3Type::Zero();
    Vector3Type             lin_acc                                                             = Vector3Type::Zero();          // could be avoided to send: is the last IMU measurment
    Vector3Type             ang_vel                                                             = Vector3Type::Zero();          // could be avoided to send: is the last IMU measurment

    Vector3Type             lin_acc_init                                                        = Vector3Type::Zero();          // value to initialize preintegration base - i.e. lin acc from predecessor
    Vector3Type             ang_vel_init                                                        = Vector3Type::Zero();          // value to initialize preintegration base - i.e. ang vel from predecessor

    TransformType           T_w_s_gt                                                            = TransformType::Identity();

    // IMU Preintegration
    PreintegrationData      preintegration;

    //Neighborhood
    LandmarksMinimalType    landmarks;
    idpair                  id_predecessor                                                      = defpair;
    idpair                  id_successor                                                        = defpair;
    idpair                  id_reference                                                        = defpair;

    // Image handling
    cv::Mat                 img;

protected:

    friend class cereal::access;                                                                                                // Serialization

    template<class Archive>
    auto save(Archive &archive) const ->void {
        if(save_to_file) {
            archive(timestamp,id,
                    calibration,
                    img_dim_x_min,img_dim_y_min,img_dim_x_max,img_dim_y_max,
                    keypoints_distorted,keypoints_undistorted,keypoints_aors,descriptors,
                    T_s_c,T_w_s,
                    velocity,bias_gyro,bias_accel,
                    lin_acc,ang_vel,
                    lin_acc_init,ang_vel_init,
                    T_w_s_gt,
                    preintegration,
                    landmarks,id_predecessor,id_successor,
                    img
                    );
        } else if(is_update_msg){
            archive(T_sref_s,id_reference,
                    is_update_msg);
        } else {
            archive(timestamp,id,
                    calibration,
                    img_dim_x_min,img_dim_y_min,img_dim_x_max,img_dim_y_max,
                    keypoints_distorted,keypoints_undistorted,keypoints_aors,descriptors,
                    T_s_c,T_sref_s,velocity,bias_gyro,bias_accel,
                    lin_acc,ang_vel,
                    lin_acc_init,ang_vel_init,
                    T_w_s_gt,
                    preintegration,
                    landmarks,id_predecessor,id_successor,id_reference,
                    is_update_msg,
                    img
                    );
        }
    }

    template<class Archive>
    auto load(Archive &archive)->void {
        if(save_to_file) {
            archive(timestamp,id,
                    calibration,
                    img_dim_x_min,img_dim_y_min,img_dim_x_max,img_dim_y_max,
                    keypoints_distorted,keypoints_undistorted,keypoints_aors,descriptors,
                    T_s_c,T_w_s,
                    velocity,bias_gyro,bias_accel,
                    lin_acc,ang_vel,
                    lin_acc_init,ang_vel_init,
                    T_w_s_gt,
                    preintegration,
                    landmarks,id_predecessor,id_successor,
                    img
                    );
        }else if(msg_type[1] == true){
            archive(T_sref_s,id_reference,
                    is_update_msg);
        } else {
            archive(timestamp,id,
                    calibration,
                    img_dim_x_min,img_dim_y_min,img_dim_x_max,img_dim_y_max,
                    keypoints_distorted,keypoints_undistorted,keypoints_aors,descriptors,
                    T_s_c,T_sref_s,velocity,bias_gyro,bias_accel,
                    lin_acc,ang_vel,
                    lin_acc_init,ang_vel_init,
                    T_w_s_gt,
                    preintegration,
                    landmarks,id_predecessor,id_successor,id_reference,
                    is_update_msg,
                    img
                    );
        }
    }
};


} //end ns

namespace cereal {

//save and load function for Eigen::Matrix type

    template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    inline
    typename std::enable_if<traits::is_output_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    save(Archive& ar, const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& matrix) {
        const std::int32_t rows = static_cast<std::int32_t>(matrix.rows());
        const std::int32_t cols = static_cast<std::int32_t>(matrix.cols());
        ar(rows);
        ar(cols);
        ar(binary_data(matrix.data(), rows * cols * sizeof(_Scalar)));
    }

    template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    inline
    typename std::enable_if<traits::is_input_serializable<BinaryData<_Scalar>, Archive>::value, void>::type
    load(Archive& ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& matrix) {
        std::int32_t rows;
        std::int32_t cols;
        ar(rows);
        ar(cols);

        matrix.resize(rows, cols);

        ar(binary_data(matrix.data(), static_cast<std::size_t>(rows * cols * sizeof(_Scalar))));
    }

//save and load function for cv::Mat type
    template<class Archive>
    inline
    void save(Archive& ar, const cv::Mat& mat) {
        int rows, cols, type;
        bool continuous;

        rows = mat.rows;
        cols = mat.cols;
        type = mat.type();
        continuous = mat.isContinuous();

        ar & rows & cols & type & continuous;

        if (continuous) {
            const int data_size = rows * cols * static_cast<int>(mat.elemSize());
            auto mat_data = cereal::binary_data(mat.ptr(), data_size);
            ar & mat_data;
        }
        else {
            const int row_size = cols * static_cast<int>(mat.elemSize());
            for (int i = 0; i < rows; i++) {
                auto row_data = cereal::binary_data(mat.ptr(i), row_size);
                ar & row_data;
            }
        }
    }

    template<class Archive>
    void load(Archive& ar, cv::Mat& mat) {
        int rows, cols, type;
        bool continuous;

        ar & rows & cols & type & continuous;

        if (continuous) {
            mat.create(rows, cols, type);
            const int data_size = rows * cols * static_cast<int>(mat.elemSize());
            auto mat_data = cereal::binary_data(mat.ptr(), data_size);
            ar & mat_data;
        }
        else {
            mat.create(rows, cols, type);
            const int row_size = cols * static_cast<int>(mat.elemSize());
            for (int i = 0; i < rows; i++) {
                auto row_data = cereal::binary_data(mat.ptr(i), row_size);
                ar & row_data;
            }
        }
    }

} /* namespace cereal */
