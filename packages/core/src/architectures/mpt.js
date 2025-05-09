// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "mpt",
  models: [
    {
      model_id: "Xenova/ipt-350m",
      dtype: "quantized",
      architectures: ["MPTForCausalLM"],
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
        "If",
        "Less",
        "MatMul",
        "MatMulInteger",
        "Mul",
        "Not",
        "Or",
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
      model_id: "Xenova/ipt-350m",
      dtype: "fp32",
      architectures: ["MPTForCausalLM"],
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
        "If",
        "Less",
        "MatMul",
        "Mul",
        "Not",
        "Or",
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
      model_id: "hf-internal-testing/tiny-random-MptForCausalLM",
      dtype: "fp32",
      architectures: ["MptForCausalLM"],
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
      model_id:
        "onnx-internal-testing/tiny-random-MptForSequenceClassification-ONNX",
      dtype: "fp32",
      architectures: ["MptForSequenceClassification"],
      ops: [
        "Add",
        "ArgMax",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Flatten",
        "Gather",
        "Identity",
        "Less",
        "MatMul",
        "Mod",
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
