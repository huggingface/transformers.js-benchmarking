// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "mobilenet_v2",
  models: [
    {
      model_id: "onnx-internal-testing/tiny-random-MobileNetV2Model-ONNX",
      dtype: "fp32",
      ops: [
        "Add",
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
    {
      model_id:
        "onnx-internal-testing/tiny-random-MobileNetV2ForImageClassification-ONNX",
      dtype: "fp32",
      ops: [
        "Add",
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
  ],
};
