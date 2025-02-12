// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "convnextv2",
  models: [
    {
      model_id: "onnx-internal-testing/tiny-random-ConvNextV2Model-ONNX",
      dtype: "fp32",
      architectures: ["ConvNextV2Model"],
      ops: [
        "Add",
        "Cast",
        "Constant",
        "Conv",
        "Div",
        "Erf",
        "Identity",
        "MatMul",
        "Mul",
        "Pow",
        "ReduceL2",
        "ReduceMean",
        "Sqrt",
        "Sub",
        "Transpose",
      ],
    },
    {
      model_id:
        "onnx-internal-testing/tiny-random-ConvNextV2ForImageClassification-ONNX",
      dtype: "fp32",
      architectures: ["ConvNextV2ForImageClassification"],
      ops: [
        "Add",
        "Cast",
        "Constant",
        "Conv",
        "Div",
        "Erf",
        "Gemm",
        "Identity",
        "MatMul",
        "Mul",
        "Pow",
        "ReduceL2",
        "ReduceMean",
        "Sqrt",
        "Sub",
        "Transpose",
      ],
    },
  ],
};
