import tensorflow as tf
import os
def GPU_Name():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    tf.test.is_built_with_cuda()
    tf.sysconfig.get_build_info()
    var = tf.sysconfig.get_build_info()["cuda_version"]
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)
GPU_Name()