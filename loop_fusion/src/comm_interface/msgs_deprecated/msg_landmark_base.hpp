#pragma once

#include <eigen3/Eigen/Eigen>

// CoVINS
#include "../typedefs.hpp"

namespace comm_interface {

struct MsgLandmarkBase {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using idpair                        = TypeDefs::idpair;
    using TransformType                 = TypeDefs::TransformType;
    using Vector3Type                   = TypeDefs::Vector3Type;

    using ObservationsMinimalType       = std::map<idpair,int>;
    using MsgTypeVector                 = TypeDefs::MsgInfoType;

    struct compare_less{bool operator() (const MsgLandmarkBase &a, const MsgLandmarkBase &b) const;};

public:

    MsgLandmarkBase();
    MsgLandmarkBase(bool filesave);
    MsgLandmarkBase(MsgTypeVector msgtype);

    // Interfaces
    auto SetMsgType(int msg_size)                                       ->void;
    auto SetMsgType(MsgTypeVector msgtype)                              ->void;

    // Infrastructure
    MsgTypeVector           msg_type                                                            = std::vector<uint32_t>(5);     // size, is_update, ID of Keyframe, ???,;
    bool                    is_update_msg                                                       = false;
    bool                    save_to_file                                                        = false;                        // indicates that this LM will be saved to a file, not send over network

    // Identifier
    idpair                  id;

    // Position
    Vector3Type             pos_ref;
    Vector3Type             pos_w;

    //Neighborhood
    ObservationsMinimalType observations;
    idpair id_reference;

protected:

    friend class cereal::access;                                                                                                // Serialization

    template<class Archive>
    auto save(Archive &archive) const ->void {
        if(save_to_file) {
            archive(id,
                    pos_w,
                    observations,id_reference
                    );
        } else if(is_update_msg){
            archive(pos_ref,id_reference,
                    is_update_msg);
        } else {
            archive(id,
                    pos_ref,
                    observations,id_reference,
                    is_update_msg);
        }
    }

    template<class Archive>
    auto load(Archive &archive)->void {
        if(save_to_file) {
            archive(id,
                    pos_w,
                    observations,id_reference
                    );
        } else if(msg_type[1] == true){
            archive(pos_ref,id_reference,
                    is_update_msg);
        } else {
            archive(id,
                    pos_ref,
                    observations,id_reference,
                    is_update_msg);
        }
    }
};

} //end ns
