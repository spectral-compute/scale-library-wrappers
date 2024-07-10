include(../common.cmake)


add_cosplay_lib(
    NAME cudnn
    MACRO_NAME "GPUDNN"
)

add_cosplay_lib(
    NAME cudnn_adv_infer
    MACRO_NAME "GPUDNN_ADV_INFER"
)
add_cosplay_lib(
    NAME cudnn_adv_train
    MACRO_NAME "GPUDNN_ADV_TRAIN"
)

add_cosplay_lib(
    NAME cudnn_cnn_train
    MACRO_NAME "GPUDNN_CNN_TRAIN"
)
add_cosplay_lib(
    NAME cudnn_cnn_infer
    MACRO_NAME "GPUDNN_CNN_INFER"
)

add_cosplay_lib(
    NAME cudnn_ops_infer
    MACRO_NAME "GPUDNN_OPS_INFER"
)
add_cosplay_lib(
    NAME cudnn_ops_train
    MACRO_NAME "GPUDNN_OPS_TRAIN"
)
