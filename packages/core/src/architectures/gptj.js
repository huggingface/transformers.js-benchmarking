// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "gptj",
  models: [
    {
      model_id: "Xenova/kogpt-j-350m",
      dtype: "quantized",
      architectures: ["GPTJForCausalLM"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "ConcatFromSequence",
        "Constant",
        "ConstantOfShape",
        "DequantizeLinear",
        "Div",
        "DynamicQuantizeLinear",
        "Equal",
        "Expand",
        "Gather",
        "GatherElements",
        "If",
        "Loop",
        "MatMul",
        "MatMulInteger",
        "Mul",
        "Neg",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "SequenceAt",
        "SequenceEmpty",
        "SequenceInsert",
        "Shape",
        "Slice",
        "Softmax",
        "Split",
        "SplitToSequence",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tanh",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "Xenova/kogpt-j-350m",
      dtype: "fp32",
      architectures: ["GPTJForCausalLM"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "ConcatFromSequence",
        "Constant",
        "ConstantOfShape",
        "Div",
        "Equal",
        "Expand",
        "Gather",
        "GatherElements",
        "If",
        "Loop",
        "MatMul",
        "Mul",
        "Neg",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "SequenceAt",
        "SequenceEmpty",
        "SequenceInsert",
        "Shape",
        "Slice",
        "Softmax",
        "Split",
        "SplitToSequence",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tanh",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "hf-internal-testing/tiny-random-GPTJForCausalLM",
      dtype: "fp32",
      architectures: ["GPTJForCausalLM"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Div",
        "Expand",
        "Gather",
        "GatherElements",
        "Identity",
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
        "Squeeze",
        "Sub",
        "Tanh",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-internal-testing/tiny-random-GPTJForCausalLM-ONNX",
      dtype: "fp32",
      architectures: ["GPTJForCausalLM"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Div",
        "Equal",
        "Expand",
        "Gather",
        "GatherElements",
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
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tanh",
        "Tile",
        "Transpose",
        "Trilu",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id:
        "onnx-internal-testing/tiny-random-GPTJForSequenceClassification-ONNX",
      dtype: "fp32",
      architectures: ["GPTJForSequenceClassification"],
      ops: [
        "Add",
        "ArgMax",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Div",
        "Equal",
        "Expand",
        "Flatten",
        "Gather",
        "GatherElements",
        "Greater",
        "Identity",
        "MatMul",
        "Mod",
        "Mul",
        "Neg",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "ScatterND",
        "Shape",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tanh",
        "Tile",
        "Transpose",
        "Trilu",
        "Unsqueeze",
        "Where",
      ],
    },
  ],
};
