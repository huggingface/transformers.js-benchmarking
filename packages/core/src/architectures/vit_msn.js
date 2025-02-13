// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "vit_msn",
  models: [
    {
      model_id: "hf-internal-testing/tiny-random-ViTMSNForImageClassification",
      dtype: "fp32",
      architectures: ["ViTMSNForImageClassification"],
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
        "Sqrt",
        "Sub",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id:
        "onnx-internal-testing/tiny-random-ViTMSNForImageClassification-ONNX",
      dtype: "fp32",
      architectures: ["ViTMSNForImageClassification"],
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
        "Gemm",
        "Identity",
        "MatMul",
        "Mul",
        "Pow",
        "ReduceMean",
        "Reshape",
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
      model_id: "onnx-internal-testing/tiny-random-ViTMSNModel-ONNX",
      dtype: "fp32",
      architectures: ["ViTMSNModel"],
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
        "Mul",
        "Pow",
        "ReduceMean",
        "Reshape",
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
