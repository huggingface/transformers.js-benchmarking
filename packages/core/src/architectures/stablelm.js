// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "stablelm",
  models: [
    {
      model_id: "Xenova/tiny-random-StableLmForCausalLM",
      dtype: "fp32",
      architectures: ["StableLmForCausalLM"],
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
        "Neg",
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
      model_id: "Xenova/tiny-random-StableLmForCausalLM",
      dtype: "quantized",
      architectures: ["StableLmForCausalLM"],
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
        "Identity",
        "Less",
        "MatMul",
        "MatMulInteger",
        "Mul",
        "Neg",
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
