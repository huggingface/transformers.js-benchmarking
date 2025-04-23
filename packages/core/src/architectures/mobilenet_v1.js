// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "mobilenet_v1",
  models: [
    {
      model_id:
        "onnx-internal-testing/tiny-random-MobileNetV1ForImageClassification-ONNX",
      dtype: "fp32",
      architectures: ["MobileNetV1ForImageClassification"],
      ops: [
        "Cast",
        "Clip",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "Flatten",
        "Gemm",
        "GlobalAveragePool",
        "Identity",
        "Pad",
        "Reshape",
        "Slice",
        "Transpose",
      ],
    },
    {
      model_id: "onnx-internal-testing/tiny-random-MobileNetV1Model-ONNX",
      dtype: "fp32",
      architectures: ["MobileNetV1Model"],
      ops: [
        "Cast",
        "Clip",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "Identity",
        "Pad",
        "Reshape",
        "Slice",
        "Transpose",
      ],
    },
  ],
};
