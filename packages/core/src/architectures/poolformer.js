// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "poolformer",
  models: [
    {
      model_id: "onnx-internal-testing/tiny-random-PoolFormerModel-ONNX",
      dtype: "fp32",
      ops: [
        "Add",
        "AveragePool",
        "Constant",
        "Conv",
        "Div",
        "Erf",
        "Identity",
        "InstanceNormalization",
        "Mul",
        "Reshape",
        "Shape",
        "Sub",
      ],
    },
    {
      model_id:
        "onnx-internal-testing/tiny-random-PoolFormerForImageClassification-ONNX",
      dtype: "fp32",
      ops: [
        "Add",
        "AveragePool",
        "Constant",
        "Conv",
        "Div",
        "Erf",
        "Gemm",
        "Identity",
        "InstanceNormalization",
        "Mul",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Sub",
      ],
    },
  ],
};
