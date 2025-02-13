// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "bert",
  models: [
    {
      model_id: "hf-internal-testing/tiny-random-BertModel",
      dtype: "fp32",
      architectures: ["BertModel"],
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
      model_id: "hf-internal-testing/tiny-random-BertForSequenceClassification",
      dtype: "fp32",
      architectures: ["BertForSequenceClassification"],
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
    {
      model_id: "hf-internal-testing/tiny-random-BertForQuestionAnswering",
      dtype: "fp32",
      architectures: ["BertForQuestionAnswering"],
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
      model_id: "onnx-internal-testing/tiny-random-BertModel-ONNX",
      dtype: "fp32",
      architectures: ["BertModel"],
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
        "Where",
      ],
    },
    {
      model_id:
        "onnx-internal-testing/tiny-random-BertForSequenceClassification-ONNX",
      dtype: "fp32",
      architectures: ["BertForSequenceClassification"],
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
        "Where",
      ],
    },
    {
      model_id:
        "onnx-internal-testing/tiny-random-BertForQuestionAnswering-ONNX",
      dtype: "fp32",
      architectures: ["BertForQuestionAnswering"],
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
        "Where",
      ],
    },
    {
      model_id: "onnx-internal-testing/tiny-random-BertForMultipleChoice-ONNX",
      dtype: "fp32",
      architectures: ["BertForMultipleChoice"],
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
        "Where",
      ],
    },
  ],
};
