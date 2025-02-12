// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "gemma",
  models: [
    {
      model_id: "Xenova/tiny-random-GemmaForCausalLM",
      dtype: "uint8",
      architectures: ["GemmaForCausalLM"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Cos",
        "DequantizeLinear",
        "Div",
        "DynamicQuantizeLinear",
        "Equal",
        "Expand",
        "Gather",
        "Greater",
        "Identity",
        "MatMul",
        "MatMulInteger",
        "Mul",
        "Neg",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "ScatterND",
        "Shape",
        "Sin",
        "Slice",
        "Softmax",
        "Sqrt",
        "Tanh",
        "Transpose",
        "Trilu",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "Xenova/tiny-random-GemmaForCausalLM",
      dtype: "fp32",
      architectures: ["GemmaForCausalLM"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Cos",
        "Div",
        "Equal",
        "Expand",
        "Gather",
        "Greater",
        "Identity",
        "MatMul",
        "Mul",
        "Neg",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "ScatterND",
        "Shape",
        "Sin",
        "Slice",
        "Softmax",
        "Sqrt",
        "Tanh",
        "Transpose",
        "Trilu",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "Xenova/tiny-random-GemmaForCausalLM",
      dtype: "bnb4",
      architectures: ["GemmaForCausalLM"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Cos",
        "Div",
        "Equal",
        "Expand",
        "Gather",
        "Greater",
        "Identity",
        "MatMul",
        "MatMulBnb4",
        "Mul",
        "Neg",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "ScatterND",
        "Shape",
        "Sin",
        "Slice",
        "Softmax",
        "Sqrt",
        "Tanh",
        "Transpose",
        "Trilu",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "Xenova/tiny-random-GemmaForCausalLM",
      dtype: "q4",
      architectures: ["GemmaForCausalLM"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Cos",
        "Div",
        "Equal",
        "Expand",
        "Gather",
        "Greater",
        "Identity",
        "MatMul",
        "MatMulNBits",
        "Mul",
        "Neg",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "ScatterND",
        "Shape",
        "Sin",
        "Slice",
        "Softmax",
        "Sqrt",
        "Tanh",
        "Transpose",
        "Trilu",
        "Unsqueeze",
        "Where",
      ],
    },
  ],
};
