// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "convnext",
  models: [
    {
      model_id: "onnx-internal-testing/tiny-random-ConvNextModel-ONNX",
      dtype: "fp32",
      architectures: ["ConvNextModel"],
      ops: [
        "Add",
        "Cast",
        "Constant",
        "Conv",
        "Div",
        "Erf",
        "Identity",
        "MatMul",
        "Mul",
        "Pow",
        "ReduceMean",
        "Sqrt",
        "Sub",
        "Transpose",
      ],
    },
    {
      model_id:
        "onnx-internal-testing/tiny-random-ConvNextForImageClassification-ONNX",
      dtype: "fp32",
      architectures: ["ConvNextForImageClassification"],
      ops: [
        "Add",
        "Cast",
        "Constant",
        "Conv",
        "Div",
        "Erf",
        "Gemm",
        "Identity",
        "MatMul",
        "Mul",
        "Pow",
        "ReduceMean",
        "Sqrt",
        "Sub",
        "Transpose",
      ],
    },
  ],
};
