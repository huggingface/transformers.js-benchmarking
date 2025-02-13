// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "mbart",
  models: [
    {
      model_id: "onnx-internal-testing/tiny-random-MBartForCausalLM-ONNX",
      dtype: "fp32",
      architectures: ["MBartForCausalLM"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Identity",
        "Less",
        "MatMul",
        "Mul",
        "Pow",
        "Range",
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
      model_id: "onnx-internal-testing/tiny-random-MBartModel-ONNX",
      dtype: "fp32",
      architectures: ["MBartModel"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Identity",
        "If",
        "Less",
        "MatMul",
        "Mul",
        "Pow",
        "Range",
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
        "onnx-internal-testing/tiny-random-MBartForSequenceClassification-ONNX",
      dtype: "fp32",
      architectures: ["MBartForSequenceClassification"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "GatherElements",
        "GatherND",
        "Gemm",
        "Identity",
        "Less",
        "MatMul",
        "Mul",
        "NonZero",
        "Not",
        "Pow",
        "Range",
        "ReduceMean",
        "ReduceSum",
        "Reshape",
        "ScatterND",
        "Shape",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tanh",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id:
        "onnx-internal-testing/tiny-random-MBartForQuestionAnswering-ONNX",
      dtype: "fp32",
      architectures: ["MBartForQuestionAnswering"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "GatherElements",
        "Identity",
        "Less",
        "MatMul",
        "Mul",
        "Not",
        "Pow",
        "Range",
        "ReduceMean",
        "ReduceSum",
        "Reshape",
        "ScatterND",
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
  ],
};
