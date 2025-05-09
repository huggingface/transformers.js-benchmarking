// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "swin",
  models: [
    {
      model_id: "Xenova/swin-tiny-patch4-window7-224",
      dtype: "quantized",
      architectures: ["SwinForImageClassification"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "ConvInteger",
        "Div",
        "DynamicQuantizeLinear",
        "Equal",
        "Erf",
        "Expand",
        "Flatten",
        "Gather",
        "GlobalAveragePool",
        "MatMul",
        "MatMulInteger",
        "Mod",
        "Mul",
        "Not",
        "Pad",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "ScatterND",
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
      model_id: "Xenova/swin-tiny-patch4-window7-224",
      dtype: "fp32",
      architectures: ["SwinForImageClassification"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Flatten",
        "Gather",
        "Gemm",
        "GlobalAveragePool",
        "MatMul",
        "Mod",
        "Mul",
        "Not",
        "Pad",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "ScatterND",
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
      model_id: "onnx-community/BiRefNet_lite-ONNX",
      dtype: "fp32",
      architectures: [],
      ops: [
        "Add",
        "BatchNormalization",
        "Cast",
        "Clip",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Floor",
        "Gather",
        "GatherND",
        "GlobalAveragePool",
        "Identity",
        "LayerNormalization",
        "MatMul",
        "Mod",
        "Mul",
        "Not",
        "Pad",
        "Relu",
        "Reshape",
        "Resize",
        "ScatterND",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Split",
        "Sub",
        "Sum",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/MVANet-ONNX",
      dtype: "q4",
      architectures: [],
      ops: [
        "Add",
        "AveragePool",
        "Concat",
        "Conv",
        "Div",
        "Erf",
        "Gather",
        "Gemm",
        "MatMul",
        "MatMulNBits",
        "Mul",
        "PRelu",
        "Pad",
        "Pow",
        "ReduceMean",
        "Relu",
        "Reshape",
        "Resize",
        "Sigmoid",
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
      model_id: "onnx-community/MVANet-ONNX",
      dtype: "bnb4",
      architectures: [],
      ops: [
        "Add",
        "AveragePool",
        "Concat",
        "Conv",
        "Div",
        "Erf",
        "Gather",
        "Gemm",
        "MatMul",
        "MatMulBnb4",
        "Mul",
        "PRelu",
        "Pad",
        "Pow",
        "ReduceMean",
        "Relu",
        "Reshape",
        "Resize",
        "Sigmoid",
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
      model_id: "onnx-community/MVANet-ONNX",
      dtype: "q4f16",
      architectures: [],
      ops: [
        "Add",
        "AveragePool",
        "Cast",
        "Concat",
        "Conv",
        "Div",
        "Erf",
        "Gather",
        "Gemm",
        "MatMul",
        "MatMulNBits",
        "Mul",
        "PRelu",
        "Pad",
        "Pow",
        "ReduceMean",
        "Relu",
        "Reshape",
        "Resize",
        "Sigmoid",
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
      model_id: "onnx-community/MVANet-ONNX",
      dtype: "quantized",
      architectures: [],
      ops: [
        "Add",
        "AveragePool",
        "Cast",
        "Concat",
        "ConvInteger",
        "Div",
        "DynamicQuantizeLinear",
        "Erf",
        "Gather",
        "MatMul",
        "MatMulInteger",
        "Mul",
        "PRelu",
        "Pad",
        "Pow",
        "ReduceMean",
        "Relu",
        "Reshape",
        "Resize",
        "Sigmoid",
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
      model_id: "onnx-community/MVANet-ONNX",
      dtype: "fp32",
      architectures: [],
      ops: [
        "Add",
        "AveragePool",
        "Concat",
        "Conv",
        "Div",
        "Erf",
        "Gather",
        "Gemm",
        "MatMul",
        "Mul",
        "PRelu",
        "Pad",
        "Pow",
        "ReduceMean",
        "Relu",
        "Reshape",
        "Resize",
        "Sigmoid",
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
      model_id: "onnx-community/MVANet-ONNX",
      dtype: "fp16",
      architectures: [],
      ops: [
        "Add",
        "AveragePool",
        "Cast",
        "Concat",
        "Conv",
        "Div",
        "Erf",
        "Gather",
        "Gemm",
        "MatMul",
        "Mul",
        "PRelu",
        "Pad",
        "Pow",
        "ReduceMean",
        "Relu",
        "Reshape",
        "Resize",
        "Sigmoid",
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
        "onnx-internal-testing/tiny-random-SwinForImageClassification-ONNX",
      dtype: "fp32",
      architectures: ["SwinForImageClassification"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Flatten",
        "Gather",
        "Gemm",
        "GlobalAveragePool",
        "Identity",
        "MatMul",
        "Mod",
        "Mul",
        "Not",
        "Pad",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "ScatterND",
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
      model_id: "onnx-internal-testing/tiny-random-SwinModel-ONNX",
      dtype: "fp32",
      architectures: ["SwinModel"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Identity",
        "MatMul",
        "Mod",
        "Mul",
        "Not",
        "Pad",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "ScatterND",
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
  ],
};
