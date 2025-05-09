// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "wavlm",
  models: [
    {
      model_id: "Xenova/wavlm-base-plus",
      dtype: "quantized",
      architectures: ["WavLMModel"],
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
        "Erf",
        "Expand",
        "Gather",
        "Greater",
        "InstanceNormalization",
        "Less",
        "Log",
        "MatMul",
        "MatMulInteger",
        "Min",
        "Mul",
        "Pow",
        "Range",
        "ReduceMean",
        "ReduceSum",
        "Reshape",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Sqrt",
        "Sub",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "Xenova/wavlm-base-plus",
      dtype: "fp32",
      architectures: ["WavLMModel"],
      ops: [
        "Abs",
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "Div",
        "Erf",
        "Expand",
        "Gather",
        "Gemm",
        "Greater",
        "InstanceNormalization",
        "Less",
        "Log",
        "MatMul",
        "Min",
        "Mul",
        "Pow",
        "Range",
        "ReduceMean",
        "ReduceSum",
        "Reshape",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Sqrt",
        "Sub",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "Xenova/wavlm-base-plus-sv",
      dtype: "quantized",
      architectures: ["WavLMForXVector"],
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
        "Erf",
        "Expand",
        "Gather",
        "Greater",
        "InstanceNormalization",
        "Less",
        "Log",
        "MatMul",
        "MatMulInteger",
        "Min",
        "Mul",
        "Pad",
        "Pow",
        "Range",
        "ReduceMean",
        "ReduceProd",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Sqrt",
        "Sub",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "Xenova/wavlm-base-plus-sv",
      dtype: "fp32",
      architectures: ["WavLMForXVector"],
      ops: [
        "Abs",
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "Div",
        "Erf",
        "Expand",
        "Gather",
        "Gemm",
        "Greater",
        "InstanceNormalization",
        "Less",
        "Log",
        "MatMul",
        "Min",
        "Mul",
        "Pad",
        "Pow",
        "Range",
        "ReduceMean",
        "ReduceProd",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Sqrt",
        "Sub",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "Xenova/wavlm-large",
      dtype: "quantized",
      architectures: ["WavLMModel"],
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
        "Erf",
        "Expand",
        "Gather",
        "Greater",
        "Less",
        "Log",
        "MatMul",
        "MatMulInteger",
        "Min",
        "Mul",
        "Pow",
        "Range",
        "ReduceMean",
        "ReduceSum",
        "Reshape",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Sqrt",
        "Sub",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "Xenova/wavlm-large",
      dtype: "fp32",
      architectures: ["WavLMModel"],
      ops: [
        "Abs",
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "Div",
        "Erf",
        "Expand",
        "Gather",
        "Gemm",
        "Greater",
        "Less",
        "Log",
        "MatMul",
        "Min",
        "Mul",
        "Pow",
        "Range",
        "ReduceMean",
        "ReduceSum",
        "Reshape",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Sqrt",
        "Sub",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-internal-testing/tiny-random-WavLMForXVector-ONNX",
      dtype: "fp32",
      architectures: ["WavLMForXVector"],
      ops: [
        "Abs",
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "Div",
        "Erf",
        "Expand",
        "Gather",
        "Gemm",
        "Greater",
        "Identity",
        "InstanceNormalization",
        "Less",
        "Log",
        "MatMul",
        "Min",
        "Mul",
        "Pow",
        "Range",
        "ReduceMean",
        "ReduceProd",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Sqrt",
        "Sub",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-internal-testing/tiny-random-WavLMModel-ONNX",
      dtype: "fp32",
      architectures: ["WavLMModel"],
      ops: [
        "Abs",
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "Div",
        "Erf",
        "Expand",
        "Gather",
        "Gemm",
        "Greater",
        "Identity",
        "InstanceNormalization",
        "Less",
        "Log",
        "MatMul",
        "Min",
        "Mul",
        "Pow",
        "Range",
        "ReduceMean",
        "ReduceSum",
        "Reshape",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Sqrt",
        "Sub",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
  ],
};
