// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "siglip",
  models: [
    {
      model_id: "Marqo/marqo-fashionSigLIP",
      dtype: "uint8",
      architectures: [],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "ConvInteger",
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
      model_id: "Marqo/marqo-fashionSigLIP",
      dtype: "fp32",
      architectures: [],
      ops: [
        "Add",
        "Concat",
        "Conv",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Gemm",
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
      model_id: "Marqo/marqo-fashionSigLIP",
      dtype: "q4f16",
      architectures: [],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Conv",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Gemm",
        "MatMul",
        "MatMulNBits",
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
      model_id: "Marqo/marqo-fashionSigLIP",
      dtype: "bnb4",
      architectures: [],
      ops: [
        "Add",
        "Concat",
        "Conv",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Gemm",
        "MatMul",
        "MatMulBnb4",
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
      model_id: "Marqo/marqo-fashionSigLIP",
      dtype: "q4",
      architectures: [],
      ops: [
        "Add",
        "Concat",
        "Conv",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Gemm",
        "MatMul",
        "MatMulNBits",
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
      model_id: "Marqo/marqo-fashionSigLIP",
      dtype: "fp16",
      architectures: [],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Conv",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Gemm",
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
      model_id: "Xenova/siglip-base-patch16-224",
      dtype: "fp32",
      architectures: ["SiglipModel"],
      ops: [
        "Abs",
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "Div",
        "Exp",
        "Expand",
        "Gather",
        "Gemm",
        "MatMul",
        "Mod",
        "Mul",
        "Pow",
        "ReduceMean",
        "ReduceSum",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tanh",
        "Tile",
        "Transpose",
        "Unsqueeze",
      ],
    },
    {
      model_id: "Xenova/siglip-base-patch16-224",
      dtype: "quantized",
      architectures: ["SiglipModel"],
      ops: [
        "Abs",
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "ConvInteger",
        "DequantizeLinear",
        "Div",
        "DynamicQuantizeLinear",
        "Exp",
        "Expand",
        "Gather",
        "MatMul",
        "MatMulInteger",
        "Mod",
        "Mul",
        "Pow",
        "ReduceMean",
        "ReduceSum",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tanh",
        "Tile",
        "Transpose",
        "Unsqueeze",
      ],
    },
    {
      model_id: "Xenova/siglip-large-patch16-256",
      dtype: "fp16",
      architectures: ["SiglipModel"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "Div",
        "Expand",
        "Gather",
        "Gemm",
        "MatMul",
        "Mod",
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
        "Tile",
        "Transpose",
        "Unsqueeze",
      ],
    },
    {
      model_id: "onnx-community/siglip2-base-patch16-256-ONNX",
      dtype: "uint8",
      architectures: [],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "ConvInteger",
        "DequantizeLinear",
        "Div",
        "DynamicQuantizeLinear",
        "Gather",
        "MatMul",
        "MatMulInteger",
        "Mul",
        "Pow",
        "ReduceL2",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tanh",
        "Tile",
        "Transpose",
        "Unsqueeze",
      ],
    },
    {
      model_id: "onnx-community/siglip2-base-patch16-256-ONNX",
      dtype: "q4",
      architectures: [],
      ops: [
        "Add",
        "Concat",
        "Conv",
        "Div",
        "Gather",
        "Gemm",
        "MatMul",
        "MatMulNBits",
        "Mul",
        "Pow",
        "ReduceL2",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tanh",
        "Tile",
        "Transpose",
        "Unsqueeze",
      ],
    },
    {
      model_id: "onnx-community/siglip2-base-patch16-256-ONNX",
      dtype: "fp32",
      architectures: [],
      ops: [
        "Add",
        "Concat",
        "Conv",
        "Div",
        "Gather",
        "Gemm",
        "MatMul",
        "Mul",
        "Pow",
        "ReduceL2",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tanh",
        "Tile",
        "Transpose",
        "Unsqueeze",
      ],
    },
    {
      model_id: "onnx-community/siglip2-base-patch16-256-ONNX",
      dtype: "q4f16",
      architectures: [],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Conv",
        "Div",
        "Gather",
        "Gemm",
        "MatMul",
        "MatMulNBits",
        "Mul",
        "Pow",
        "ReduceL2",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tanh",
        "Tile",
        "Transpose",
        "Unsqueeze",
      ],
    },
    {
      model_id: "onnx-community/siglip2-base-patch16-256-ONNX",
      dtype: "bnb4",
      architectures: [],
      ops: [
        "Add",
        "Concat",
        "Conv",
        "Div",
        "Gather",
        "Gemm",
        "MatMul",
        "MatMulBnb4",
        "Mul",
        "Pow",
        "ReduceL2",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tanh",
        "Tile",
        "Transpose",
        "Unsqueeze",
      ],
    },
    {
      model_id: "onnx-community/siglip2-base-patch16-256-ONNX",
      dtype: "fp16",
      architectures: [],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Conv",
        "Div",
        "Gather",
        "Gemm",
        "MatMul",
        "Mul",
        "Pow",
        "ReduceL2",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tanh",
        "Tile",
        "Transpose",
        "Unsqueeze",
      ],
    },
    {
      model_id: "onnx-internal-testing/tiny-random-SiglipModel-ONNX",
      dtype: "fp32",
      architectures: ["SiglipModel"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "Div",
        "Exp",
        "Expand",
        "Gather",
        "Gemm",
        "Identity",
        "MatMul",
        "Mod",
        "Mul",
        "Pow",
        "ReduceL2",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tanh",
        "Tile",
        "Transpose",
        "Unsqueeze",
      ],
    },
  ],
};
