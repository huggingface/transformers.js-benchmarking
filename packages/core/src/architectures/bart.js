// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "bart",
  models: [
    {
      model_id: "hf-internal-testing/tiny-random-BartForConditionalGeneration",
      dtype: "fp32",
      architectures: ["BartForConditionalGeneration"],
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
        "onnx-internal-testing/tiny-random-BartForSequenceClassification-ONNX",
      dtype: "fp32",
      architectures: ["BartForSequenceClassification"],
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
        "GatherND",
        "Gemm",
        "Identity",
        "Less",
        "MatMul",
        "Mul",
        "NonZero",
        "Pow",
        "Range",
        "ReduceMean",
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
        "onnx-internal-testing/tiny-random-BartForQuestionAnswering-ONNX",
      dtype: "fp32",
      architectures: ["BartForQuestionAnswering"],
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
    {
      model_id: "onnx-internal-testing/tiny-random-BartForCausalLM-ONNX",
      dtype: "fp32",
      architectures: ["BartForCausalLM"],
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
  ],
};
