// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "megatron-bert",
  models: [
    {
      model_id:
        "onnx-internal-testing/tiny-random-MegatronBertForPreTraining-ONNX",
      dtype: "fp32",
      architectures: ["MegatronBertForPreTraining"],
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
        "onnx-internal-testing/tiny-random-MegatronBertForMultipleChoice-ONNX",
      dtype: "fp32",
      architectures: ["MegatronBertForMultipleChoice"],
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
        "Squeeze",
        "Sub",
        "Tanh",
        "Transpose",
        "Unsqueeze",
      ],
    },
    {
      model_id:
        "onnx-internal-testing/tiny-random-MegatronBertForQuestionAnswering-ONNX",
      dtype: "fp32",
      architectures: ["MegatronBertForQuestionAnswering"],
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
        "Split",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Transpose",
        "Unsqueeze",
      ],
    },
    {
      model_id:
        "onnx-internal-testing/tiny-random-MegatronBertForSequenceClassification-ONNX",
      dtype: "fp32",
      architectures: ["MegatronBertForSequenceClassification"],
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
