// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "ultravox",
  models: [
    {
      model_id: "onnx-community/ultravox-v0_5-llama-3_2-1b-ONNX",
      dtype: "int8",
      architectures: ["UltravoxModel"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "ConvInteger",
        "DequantizeLinear",
        "Div",
        "DynamicQuantizeLinear",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Less",
        "MatMul",
        "MatMulInteger",
        "Mul",
        "MultiHeadAttention",
        "Pad",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "RotaryEmbedding",
        "Shape",
        "Sigmoid",
        "SimplifiedLayerNormalization",
        "SkipSimplifiedLayerNormalization",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/ultravox-v0_5-llama-3_2-1b-ONNX",
      dtype: "fp32",
      architectures: ["UltravoxModel"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "Conv",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Less",
        "MatMul",
        "Mul",
        "MultiHeadAttention",
        "Pad",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "RotaryEmbedding",
        "Shape",
        "Sigmoid",
        "SimplifiedLayerNormalization",
        "SkipSimplifiedLayerNormalization",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/ultravox-v0_5-llama-3_2-1b-ONNX",
      dtype: "q4",
      architectures: ["UltravoxModel"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "Conv",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Less",
        "MatMul",
        "MatMulNBits",
        "Mul",
        "MultiHeadAttention",
        "Pad",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "RotaryEmbedding",
        "Shape",
        "Sigmoid",
        "SimplifiedLayerNormalization",
        "SkipSimplifiedLayerNormalization",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/ultravox-v0_5-llama-3_2-1b-ONNX",
      dtype: "bnb4",
      architectures: ["UltravoxModel"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "Conv",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Less",
        "MatMul",
        "MatMulBnb4",
        "Mul",
        "MultiHeadAttention",
        "Pad",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "RotaryEmbedding",
        "Shape",
        "Sigmoid",
        "SimplifiedLayerNormalization",
        "SkipSimplifiedLayerNormalization",
        "Slice",
        "Softmax",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
  ],
};
