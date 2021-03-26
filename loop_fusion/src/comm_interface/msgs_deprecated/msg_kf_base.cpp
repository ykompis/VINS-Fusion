#include "msg_kf_base.hpp"

namespace comm_interface {

MsgKeyframeBase::MsgKeyframeBase() {
    //...
}

MsgKeyframeBase::MsgKeyframeBase(bool filesave)
    : save_to_file(filesave)
{
    //...
}

MsgKeyframeBase::MsgKeyframeBase(MsgTypeVector msgtype)
    : msg_type(msgtype)
{
    //...
}

auto MsgKeyframeBase::SetMsgType(int msg_size)->void {
    msg_type[0] = msg_size;
    msg_type[1] = (int)is_update_msg;
    msg_type[2] = id.first;
    msg_type[3] = id.second;
    msg_type[4] = 0;
}

auto MsgKeyframeBase::SetMsgType(MsgTypeVector msgtype)->void {
    msg_type = msgtype;
}

} //end ns
