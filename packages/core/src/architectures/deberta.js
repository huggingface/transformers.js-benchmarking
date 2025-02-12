// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "deberta",
  models: [
    {
      model_id: "onnx-internal-testing/tiny-random-DebertaModel-ONNX",
      dtype: "fp32",
      architectures: ["DebertaModel"],
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
        "Not",
        "Pow",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id:
        "onnx-internal-testing/tiny-random-DebertaForQuestionAnswering-ONNX",
      dtype: "fp32",
      architectures: ["DebertaForQuestionAnswering"],
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
        "Not",
        "Pow",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id:
        "onnx-internal-testing/tiny-random-DebertaForSequenceClassification-ONNX",
      dtype: "fp32",
      architectures: ["DebertaForSequenceClassification"],
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
        "Not",
        "Pow",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
  ],
};
