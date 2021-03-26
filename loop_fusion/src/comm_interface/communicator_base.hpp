#pragma once

// C++
#include <unistd.h>
#include <mutex>
#include <thread>
#include <netinet/in.h>
#include <eigen3/Eigen/Eigen>

// CoVINS
#include "typedefs.hpp"
#include "msgs/msg_kf_base.hpp"
#include "msgs/msg_landmark_base.hpp"

#define ContainerSize 10

namespace comm_interface {

struct MsgKeyframeBase;
struct MsgLandmarkBase;

class Keyframe {

};

class Landmark {

};

struct data_bundle {
public:

    struct compare_less{bool operator() (const data_bundle &a, const data_bundle &b) const;};

    std::list<MsgKeyframeBase> keyframes;
    std::list<MsgLandmarkBase> landmarks;
};

struct message_container {
public:
    using MsgInfoType                   = TypeDefs::MsgInfoType;
public:
    std::stringstream                   ser_msg;
    MsgInfoType                         msg_info;
};

class CommunicatorBase {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using idpair                        = TypeDefs::idpair;
    using MsgInfoType                   = TypeDefs::MsgInfoType;

    using TransformType                 = TypeDefs::TransformType;

    using LandmarkPtr                   = std::shared_ptr<Landmark>;
    using KeyframePtr                   = std::shared_ptr<Keyframe>;

    using DataBundleBufferType          = std::list<data_bundle>;
    using KeyframeBufferType            = std::list<MsgKeyframeBase,Eigen::aligned_allocator<MsgKeyframeBase>>;
    using LandmarkBufferType            = std::list<MsgLandmarkBase,Eigen::aligned_allocator<MsgLandmarkBase>>;

public:

    CommunicatorBase();
    CommunicatorBase(int client_id, int newfd);

    // Main
    virtual auto Run()                                                          ->void                      = 0;

    // Interfaces
    virtual auto PassDataBundle(data_bundle &msg)                               ->void;
    virtual auto Lock()                                                         ->void;
    virtual auto UnLock()                                                       ->void;
    virtual auto TryLock()                                                      ->bool;
    virtual auto GetClientId()                                                  ->int;

    // Message handling
    virtual auto Serialize(MsgKeyframeBase &msg)                                ->void;
    virtual auto Serialize(MsgLandmarkBase &msg)                                ->void;

    // Message passing
    static auto GetInAddr(struct sockaddr *sa)->void*;                                                                          // get sockaddr, IPv4 or IPv6:
    virtual auto ConnectToServer(const char *node, std::string port)            ->int;

    // Synchronization
    auto SetFinish()                                                    ->void
        {std::unique_lock<std::mutex> lock(mtx_finish_); finish_ = true;}
    auto ShallFinish()                                                  ->bool
        {std::unique_lock<std::mutex> lock(mtx_finish_); return finish_;}

protected:

    // Message passing
    virtual auto SendAll(MsgInfoType &msg_send)                                 ->int;
    virtual auto SendAll(std::stringstream &msg)                                ->int;
    virtual auto RecvAll(unsigned int sz, std::vector<char> &buffer)            ->int;
    virtual auto RecvAll(unsigned int sz, MsgInfoType &buffer)                  ->int;
    virtual auto RecvMsg()                                                      ->void;
    virtual auto WriteToBuffer()                                                ->void;
    virtual auto CheckBuffer()                                                  ->bool;
    virtual auto packi32(std::vector<unsigned char> &buf, MsgInfoType &msg)     ->void;
    virtual auto unpacki32(std::vector<unsigned char> &buf, MsgInfoType &msg)   ->void;
    virtual auto SendMsgContainer(message_container &msg)                       ->void;

    // Data handling
    virtual auto ProcessBufferOut()                                             ->void;
    virtual auto ProcessBufferIn()                                              ->void;
    virtual auto ProcessKeyframeMessages()                                      ->void                      = 0;
    virtual auto ProcessLandmarkMessages()                                      ->void                      = 0;
    virtual auto ProcessNewKeyframes()                                          ->void                      = 0;
    virtual auto ProcessNewLandmarks()                                          ->void                      = 0;
    virtual auto ProcessAdditional()                                            ->void                      = 0;

    // Infrastructure
    int                                 client_id_                                                          = -1;

    // Data
    std::list<std::vector<char>>        buffer_recv_data_;
    std::list<std::vector<uint32_t>>    buffer_recv_info_;

    DataBundleBufferType                buffer_data_out_;
    KeyframeBufferType                  buffer_keyframes_out_;
    LandmarkBufferType                  buffer_landmarks_out_;

    KeyframeBufferType                  buffer_keyframes_in_;
    LandmarkBufferType                  buffer_landmarks_in_;


    // Sync
    std::mutex                          mtx_comm_;
    std::mutex                          mtx_finish_;
    std::mutex                          mtx_recv_buffer_;
    std::mutex                          mtx_out_;
    std::mutex                          mtx_in_;

    bool                                finish_                                                             = false;

    //message passing
    std::vector<char>                   recv_buf_;
    int                                 package_size_send_;
    int                                 newfd_;
    std::stringstream                   send_ser_;
    std::vector<char>                   send_buf_;                                                                              //copy shared buffer to send thread for deserialization
    MsgInfoType                         msg_type_buf_                                                       = MsgInfoType(5);   // msg size, SentOnce, id.first, id.second
    MsgInfoType                         msg_type_container_;                                                                    // msg size, SentOnce, id.first, id.second
    MsgInfoType                         msg_type_deserialize_                                               = MsgInfoType(5);   // msg size, SentOnce, id.first, id.second
};

} //end ns
