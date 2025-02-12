// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "layoutlm",
  models: [
    {
      model_id: "onnx-internal-testing/tiny-random-LayoutLMModel-ONNX",
      dtype: "fp32",
      architectures: ["LayoutLMModel"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "Div",
        "Erf",
        "Gather",
        "Identity",
        "MatMul",
        "Mul",
        "Pow",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Sqrt",
        "Sub",
        "Transpose",
        "Unsqueeze",
      ],
    },
    {
      model_id:
        "onnx-internal-testing/tiny-random-LayoutLMForSequenceClassification-ONNX",
      dtype: "fp32",
      architectures: ["LayoutLMForSequenceClassification"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "Div",
        "Erf",
        "Gather",
        "Gemm",
        "Identity",
        "MatMul",
        "Mul",
        "Pow",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Sqrt",
        "Sub",
        "Tanh",
        "Transpose",
        "Unsqueeze",
      ],
    },
  ],
};
