#pragma once

#include <thread>
#include <eigen3/Eigen/Eigen>

//SERIALIZATION
#include "cereal/cereal.hpp"
#include "cereal/types/memory.hpp"
#include "cereal/types/utility.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/types/polymorphic.hpp"
#include "cereal/types/concepts/pair_associative_container.hpp"
#include "cereal/types/base_class.hpp"
#include "cereal/archives/binary.hpp"
#include "cereal/archives/binary.hpp"
#include "cereal/access.hpp"
#include "cstdint"

#define MAPRANGE std::numeric_limits<uint8_t>::max()
#define KFRANGE std::numeric_limits<uint16_t>::max()
#define MPRANGE std::numeric_limits<uint32_t>::max()
#define UIDRANGE std::numeric_limits<uint32_t>::max()

#define defpair std::make_pair((size_t)KFRANGE,(size_t)MAPRANGE) //default pair
#define defid -1 //default id

namespace comm_interface {

struct TypeDefs {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using precision_t                   = double;
    using idpair                        = std::pair<size_t,size_t>;
    using MsgInfoType                   = std::vector<uint32_t>;

    using Vector2Type                   = Eigen::Matrix<precision_t,2,1>;
    using Vector3Type                   = Eigen::Matrix<precision_t,3,1>;
    using Vector4Type                   = Eigen::Matrix<precision_t,4,1>;
    using DynamicVectorType             = Eigen::Matrix<precision_t,Eigen::Dynamic,1>;
    using QuaternionType                = Eigen::Quaternion<precision_t>;

    using TransformType                 = Eigen::Matrix<precision_t,4,4>;
    using Matrix3Type                   = Eigen::Matrix<precision_t,3,3>;

    using Vector2Vector                 = std::vector<Vector2Type,Eigen::aligned_allocator<Vector2Type>>;
    using Vector3Vector                 = std::vector<Vector3Type,Eigen::aligned_allocator<Vector3Type>>;
    using KeypointType                  = Vector2Type;
    using KeypointVector                = std::vector<KeypointType,Eigen::aligned_allocator<KeypointType>>;
    using AorsVector                    = std::vector<Vector4Type,Eigen::aligned_allocator<Vector4Type>>;

    using ThreadPtr                     = std::unique_ptr<std::thread>;
};

inline std::ostream &operator<<(std::ostream &out, const TypeDefs::idpair id) {
    return out << id.first << "|" << id.second;
}

enum eDistortionModel
{
    NOTSET_DIST     = -1,
    RADTAN          =  0,
    EQUI            =  1,
    PLUMBBOB        =  2
};

enum eCamModel
{
    NOTSET_CAM      = -1,
    PINHOLE         =  0,
    OMNI            =  1
};

class Keyframe;

struct LoopConstraint {
    using TransformType             = TypeDefs::TransformType;
    using KeyframePtr               = std::shared_ptr<Keyframe>;

    LoopConstraint(KeyframePtr k1, KeyframePtr k2, TransformType T_12)
        : kf1(k1),kf2(k2),T_s1_s2(T_12) {}
    KeyframePtr         kf1;
    KeyframePtr         kf2;
    TransformType       T_s1_s2;
};

struct VICalibration {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using precision_t                   = TypeDefs::precision_t;
    using Vector2Type                   = TypeDefs::Vector2Type;
    using Vector3Type                   = TypeDefs::Vector3Type;
    using Matrix3Type                   = TypeDefs::Matrix3Type;
    using TransformType                 = TypeDefs::TransformType;
    using DynamicVectorType             = TypeDefs::DynamicVectorType;

public:

    VICalibration()
        : T_SC(Eigen::Matrix4d::Zero()),cam_model(static_cast<eCamModel>(-1)),dist_model(static_cast<eDistortionModel>(-1)),
          img_dims((Eigen::Matrix<double,2,1>::Zero())),
          dist_coeffs(Eigen::Vector4d::Zero()),
          intrinsics(Eigen::Matrix<double,4,1>::Zero()),
          K((Eigen::Matrix<double,3,3>() << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).finished()),
          a_max(0.0),g_max(0.0),
          sigma_a_c(0.0),sigma_g_c(0.0),sigma_ba(0.0),sigma_bg(0.0),sigma_aw_c(0.0),sigma_gw_c(0.0),
          tau(0.0),g(0.0),a0(Eigen::Vector3d::Zero()),rate(0),
          delay_cam0_to_imu(0.0),delay_cam1_to_imu(0.0)
    {}

    VICalibration(Eigen::Matrix4d Tsc,eCamModel cmodel, eDistortionModel dmodel,
                  Eigen::VectorXd DistCoeffs,
                  double dw,double dh,
                  double dfx,double dfy,double dcx,double dcy,
                  double damax,double dgmax,double dsigmaac,double dsigmagc,double dsigmaba,double dsigmabg,double dsigmaawc,double dsigmagwc,
                  double dtau,double dg,Eigen::Vector3d va0,int irate,
                  double dDelayC0toIMU,double dDelayC1toIMU)
        : T_SC(Tsc),cam_model(cmodel),dist_model(dmodel),
          img_dims((Eigen::Matrix<double,2,1>() << dw,dh).finished()),
          dist_coeffs(DistCoeffs),
          intrinsics((Eigen::Matrix<double,4,1>() << dfx,dfy,dcx,dcy).finished()),
          K((Eigen::Matrix<double,3,3>() << dfx, 0.0, dcx, 0.0, dfy, dcy, 0.0, 0.0, 1.0).finished()),
          a_max(damax),g_max(dgmax),
          sigma_a_c(dsigmaac),sigma_g_c(dsigmagc),sigma_ba(dsigmaba),sigma_bg(dsigmabg),sigma_aw_c(dsigmaawc),sigma_gw_c(dsigmagwc),
          tau(dtau),g(dg),a0(va0),rate(irate),
          delay_cam0_to_imu(dDelayC0toIMU),delay_cam1_to_imu(dDelayC1toIMU)
    {}

    //Cam
    Eigen::Matrix4d             T_SC;                                                                                           ///< Transformation from camera to sensor (IMU) frame.
    eCamModel                   cam_model;                                                                                      ///< Distortion type. ('pinhole' 'omni')
    eDistortionModel            dist_model;                                                                                     ///< Distortion type. ('radialtangential' 'plumb_bob' 'equidistant')
    Vector2Type                 img_dims;                                                                                       ///< Image dimension. [pixels] (width;height)
    DynamicVectorType           dist_coeffs;                                                                                    ///< Distortion Coefficients.
    DynamicVectorType           intrinsics;                                                                                     ///< fx fy cx cy
    Matrix3Type                 K;
    //IMU
    precision_t                 a_max;                                                                                          ///< Accelerometer saturation. [m/s^2] -- not used
    precision_t                 g_max;                                                                                          ///< Gyroscope saturation. [rad/s] -- not used
    precision_t                 sigma_a_c;                                                                                      ///< Accelerometer noise density.
    precision_t                 sigma_g_c;                                                                                      ///< Gyroscope noise density.
    precision_t                 sigma_ba;                                                                                       ///< Initial accelerometer bias -- not used
    precision_t                 sigma_bg;                                                                                       ///< Initial gyroscope bias. -- not used
    precision_t                 sigma_aw_c;                                                                                     ///< Accelerometer drift noise density.
    precision_t                 sigma_gw_c;                                                                                     ///< Gyroscope drift noise density.
    precision_t                 tau;                                                                                            ///< Reversion time constant of accerometer bias. [s] -- not used
    precision_t                 g;                                                                                              ///< Earth acceleration.
    Vector3Type                 a0;                                                                                             ///< Mean of the prior accelerometer bias. -- not used
    int                         rate;                                                                                           ///< IMU rate in Hz.
    precision_t                 delay_cam0_to_imu;                                                                              ///< Timestamp shift. Timu = Timage + image_delay
    precision_t                 delay_cam1_to_imu;                                                                              ///< Timestamp shift. Timu = Timage + image_delay

    void show()
    {
        std::cout << "--- Cam ---" << std::endl;
        std::cout << "T_SC: \n" << T_SC << std::endl;
        if(cam_model == eCamModel::PINHOLE)                 std::cout << "CamModel: pinhole" << std::endl;
        else if(cam_model == eCamModel::OMNI)               std::cout << "CamModel: omni" << std::endl;
        if(dist_model == eDistortionModel::RADTAN)          std::cout << "DistortionModel: radialtangential" << std::endl;
        else if(dist_model == eDistortionModel::EQUI)       std::cout << "DistortionModel: equidistant" << std::endl;
        else if(dist_model == eDistortionModel::PLUMBBOB)   std::cout << "DistortionModel: plumb_bob" << std::endl;
        std::cout << "Image Dimensions: \n" << img_dims << std::endl;
        std::cout << "Distortion Coefficients: \n" << dist_coeffs << std::endl;
        std::cout << "Intrinsics: \n" << intrinsics << std::endl;
        std::cout << "K: \n" << K << std::endl;
        std::cout << "--- IMU ---" << std::endl;
        std::cout << "a_max: " << a_max << std::endl;
        std::cout << "g_max: " << g_max << std::endl;
        std::cout << "sigma_g_c: " << sigma_g_c << std::endl;
        std::cout << "sigma_bg: " << sigma_bg << std::endl;
        std::cout << "sigma_a_c: " << sigma_a_c << std::endl;
        std::cout << "sigma_ba: " << sigma_ba << std::endl;
        std::cout << "sigma_gw_c: " << sigma_gw_c << std::endl;
        std::cout << "sigma_aw_c: " << sigma_aw_c << std::endl;
        std::cout << "tau: " << tau << std::endl;
        std::cout << "g: " << g << std::endl;
        std::cout << "a0: \n" << a0 << std::endl;
        std::cout << "rate: " << rate << std::endl;
        std::cout << "delay_cam0_to_imu: " << delay_cam0_to_imu << std::endl;
        std::cout << "delay_cam1_to_imu: " << delay_cam1_to_imu << std::endl;
    }

    template<class Archive> auto serialize( Archive & archive )->void {
        archive(T_SC, cam_model, dist_model, img_dims, dist_coeffs, intrinsics, K,
                a_max, g_max, sigma_a_c, sigma_g_c, sigma_ba, sigma_bg, sigma_aw_c, sigma_gw_c, tau, g, a0,
                rate, delay_cam0_to_imu, delay_cam1_to_imu);
    }
};

} //end ns
