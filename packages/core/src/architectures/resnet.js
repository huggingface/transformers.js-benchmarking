// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "resnet",
  models: [
    {
      model_id: "onnx-internal-testing/tiny-random-ResNetModel-ONNX",
      dtype: "fp32",
      architectures: ["ResNetModel"],
      ops: ["Add", "Conv", "Identity", "MaxPool", "Relu"],
    },
    {
      model_id:
        "onnx-internal-testing/tiny-random-ResNetForImageClassification-ONNX",
      dtype: "fp32",
      architectures: ["ResNetForImageClassification"],
      ops: [
        "Add",
        "Conv",
        "Flatten",
        "Gemm",
        "GlobalAveragePool",
        "Identity",
        "MaxPool",
        "Relu",
      ],
    },
  ],
};
