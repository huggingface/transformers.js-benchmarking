// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "new",
  models: [
    {
      model_id: "Alibaba-NLP/gte-base-en-v1.5",
      dtype: "q4",
      architectures: ["NewModel"],
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
        "MatMul",
        "MatMulNBits",
        "Mul",
        "Neg",
        "Pow",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "Alibaba-NLP/gte-base-en-v1.5",
      dtype: "bnb4",
      architectures: ["NewModel"],
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
        "MatMul",
        "MatMulBnb4",
        "Mul",
        "Neg",
        "Pow",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "Alibaba-NLP/gte-base-en-v1.5",
      dtype: "quantized",
      architectures: ["NewModel"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "DequantizeLinear",
        "Div",
        "DynamicQuantizeLinear",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "MatMul",
        "MatMulInteger",
        "Mul",
        "Neg",
        "Pow",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "Alibaba-NLP/gte-base-en-v1.5",
      dtype: "fp32",
      architectures: ["NewModel"],
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
        "MatMul",
        "Mul",
        "Neg",
        "Pow",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/gte-multilingual-base",
      dtype: "q4",
      architectures: ["NewModel", "NewForTokenClassification"],
      ops: [
        "Add",
        "Cast",
        "Clip",
        "Concat",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "MatMul",
        "MatMulNBits",
        "Mul",
        "Neg",
        "Pow",
        "ReduceL2",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/gte-multilingual-base",
      dtype: "bnb4",
      architectures: ["NewModel", "NewForTokenClassification"],
      ops: [
        "Add",
        "Cast",
        "Clip",
        "Concat",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "MatMul",
        "MatMulBnb4",
        "Mul",
        "Neg",
        "Pow",
        "ReduceL2",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/gte-multilingual-base",
      dtype: "quantized",
      architectures: ["NewModel", "NewForTokenClassification"],
      ops: [
        "Add",
        "Cast",
        "Clip",
        "Concat",
        "DequantizeLinear",
        "Div",
        "DynamicQuantizeLinear",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "MatMul",
        "MatMulInteger",
        "Mul",
        "Neg",
        "Pow",
        "ReduceL2",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/gte-multilingual-base",
      dtype: "fp32",
      architectures: ["NewModel", "NewForTokenClassification"],
      ops: [
        "Add",
        "Cast",
        "Clip",
        "Concat",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "MatMul",
        "Mul",
        "Neg",
        "Pow",
        "ReduceL2",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/gte-multilingual-reranker-base",
      dtype: "q4",
      architectures: ["NewForSequenceClassification"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Gemm",
        "MatMul",
        "MatMulNBits",
        "Mul",
        "Neg",
        "Pow",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Tanh",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/gte-multilingual-reranker-base",
      dtype: "bnb4",
      architectures: ["NewForSequenceClassification"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Gemm",
        "MatMul",
        "MatMulBnb4",
        "Mul",
        "Neg",
        "Pow",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Tanh",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/gte-multilingual-reranker-base",
      dtype: "quantized",
      architectures: ["NewForSequenceClassification"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "DequantizeLinear",
        "Div",
        "DynamicQuantizeLinear",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "MatMul",
        "MatMulInteger",
        "Mul",
        "Neg",
        "Pow",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Tanh",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/gte-multilingual-reranker-base",
      dtype: "fp32",
      architectures: ["NewForSequenceClassification"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Gemm",
        "MatMul",
        "Mul",
        "Neg",
        "Pow",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Tanh",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
  ],
};
