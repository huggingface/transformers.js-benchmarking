// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "paligemma",
  models: [
    {
      model_id:
        "hf-internal-testing/tiny-random-PaliGemmaForConditionalGeneration",
      dtype: "fp32",
      architectures: ["PaliGemmaForConditionalGeneration"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "Conv",
        "Cos",
        "Div",
        "Equal",
        "Expand",
        "Gather",
        "MatMul",
        "Mul",
        "Neg",
        "Pow",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Sin",
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
      model_id: "onnx-community/paligemma2-3b-ft-docci-448",
      dtype: "fp16",
      architectures: ["PaliGemmaForConditionalGeneration"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "Conv",
        "Cos",
        "Div",
        "Equal",
        "Expand",
        "Gather",
        "MatMul",
        "Mul",
        "Neg",
        "Pow",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Sin",
        "Slice",
        "Softmax",
        "Sqrt",
        "Sub",
        "Tanh",
        "Transpose",
        "Trilu",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/paligemma2-3b-ft-docci-448",
      dtype: "int8",
      architectures: ["PaliGemmaForConditionalGeneration"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "ConvInteger",
        "Cos",
        "DequantizeLinear",
        "Div",
        "DynamicQuantizeLinear",
        "Equal",
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
        "Sin",
        "Slice",
        "Softmax",
        "Sqrt",
        "Sub",
        "Tanh",
        "Transpose",
        "Trilu",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/paligemma2-3b-ft-docci-448",
      dtype: "bnb4",
      architectures: ["PaliGemmaForConditionalGeneration"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "Conv",
        "Cos",
        "Div",
        "Equal",
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
        "Sin",
        "Slice",
        "Softmax",
        "Sqrt",
        "Sub",
        "Tanh",
        "Transpose",
        "Trilu",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/paligemma2-3b-ft-docci-448",
      dtype: "q4f16",
      architectures: ["PaliGemmaForConditionalGeneration"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "Conv",
        "Cos",
        "Div",
        "Equal",
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
        "Sin",
        "Slice",
        "Softmax",
        "Sqrt",
        "Sub",
        "Tanh",
        "Transpose",
        "Trilu",
        "Unsqueeze",
        "Where",
      ],
    },
  ],
};
