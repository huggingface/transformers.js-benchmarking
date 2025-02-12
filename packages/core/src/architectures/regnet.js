// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "regnet",
  models: [
    {
      model_id:
        "onnx-internal-testing/tiny-random-RegNetForImageClassification-ONNX",
      dtype: "fp32",
      architectures: ["RegNetForImageClassification"],
      ops: [
        "Add",
        "Conv",
        "Flatten",
        "Gemm",
        "GlobalAveragePool",
        "Identity",
        "Mul",
        "Relu",
        "Sigmoid",
      ],
    },
    {
      model_id: "onnx-internal-testing/tiny-random-RegNetModel-ONNX",
      dtype: "fp32",
      architectures: ["RegNetModel"],
      ops: [
        "Add",
        "Conv",
        "GlobalAveragePool",
        "Identity",
        "Mul",
        "Relu",
        "Sigmoid",
      ],
    },
  ],
};
