import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def build_engine(onnx_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1)   # 设置network的标志
    config = builder.create_builder_config() # 配置项

    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnx_file_path, 'rb') as model:
            parsed = parser.parse(model.read())
            if not parsed:
                print("Parsing ONNX model failed. Please check if the ONNX model is compatible with the TensorRT version.")
                return None

        # Option to enable FP16 mode in TensorRT
        config.set_flag(trt.BuilderFlag.FP16) # 若需要开启半精度以提高性能，请取消注释此行

        # 设置其他参数，例如最大工作空间和最大批量大小
        config.max_workspace_size = 2**32   # Increase max_workspace_size to 4 GB
        builder.max_batch_size = 1

        return builder.build_engine(network, config)


def main():

    ONNX_MODEL_PATH = '/root/LMSCNet/weight/LMSCNet.onnx'  # Change this to where onnx model was exported
    TRT_MODEL_PATH = '/root/LMSCNet/weight/LMSCNet.trt'  # Change this to where you want to save TRT model

    # Build and save the TensorRT engine
    engine = build_engine(ONNX_MODEL_PATH)

    if engine is not None:
        with open(TRT_MODEL_PATH, 'wb') as f:
            f.write(engine.serialize())
        print(f"TensorRT model saved at: {TRT_MODEL_PATH}")
    else:
        print("Failed to create TensorRT Engine. Please check the logs for any issues.")


if __name__ == '__main__':
    main()
