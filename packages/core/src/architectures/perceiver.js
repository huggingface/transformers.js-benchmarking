// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "perceiver",
  models: [
    {
      model_id:
        "onnx-internal-testing/tiny-random-PerceiverForImageClassificationConvProcessing-ONNX",
      dtype: "fp32",
      architectures: ["PerceiverForImageClassificationConvProcessing"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Conv",
        "Cos",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Identity",
        "MatMul",
        "MaxPool",
        "Mul",
        "Pad",
        "Pow",
        "Range",
        "ReduceMean",
        "Relu",
        "Reshape",
        "Shape",
        "Sin",
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
      model_id:
        "onnx-internal-testing/tiny-random-PerceiverForImageClassificationFourier-ONNX",
      dtype: "fp32",
      architectures: ["PerceiverForImageClassificationFourier"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Cos",
        "Div",
        "Equal",
        "Erf",
        "Expand",
        "Gather",
        "Identity",
        "MatMul",
        "Mul",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "Shape",
        "Sin",
        "Softmax",
        "Sqrt",
        "Sub",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-internal-testing/tiny-random-PerceiverForMaskedLM-ONNX",
      dtype: "fp32",
      architectures: ["PerceiverForMaskedLM"],
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
        "MatMul",
        "Mul",
        "Pow",
        "Range",
        "ReduceMean",
        "Reshape",
        "Shape",
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
