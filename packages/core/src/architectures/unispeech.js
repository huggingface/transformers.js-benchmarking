// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "unispeech",
  models: [
    {
      model_id: "Xenova/unispeech-large-1500h-cv",
      dtype: "quantized",
      architectures: ["UniSpeechForPreTraining"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "ConvInteger",
        "Div",
        "DynamicQuantizeLinear",
        "Erf",
        "Gather",
        "MatMul",
        "MatMulInteger",
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
      ],
    },
    {
      model_id: "Xenova/unispeech-large-1500h-cv",
      dtype: "fp32",
      architectures: ["UniSpeechForPreTraining"],
      ops: [
        "Add",
        "Concat",
        "Constant",
        "Conv",
        "Div",
        "Erf",
        "Gather",
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
      ],
    },
    {
      model_id: "hf-internal-testing/tiny-random-unispeech",
      dtype: "fp32",
      architectures: ["UniSpeechForSequenceClassification"],
      ops: [
        "Add",
        "Concat",
        "Constant",
        "Conv",
        "Div",
        "Erf",
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
      ],
    },
    {
      model_id:
        "onnx-internal-testing/tiny-random-UniSpeechForSequenceClassification-ONNX",
      dtype: "fp32",
      architectures: ["UniSpeechForSequenceClassification"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "Conv",
        "Div",
        "Erf",
        "Gather",
        "Gemm",
        "Identity",
        "InstanceNormalization",
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
      ],
    },
    {
      model_id: "onnx-internal-testing/tiny-random-UniSpeechModel-ONNX",
      dtype: "fp32",
      architectures: ["UniSpeechModel"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "Conv",
        "Div",
        "Erf",
        "Gather",
        "Identity",
        "InstanceNormalization",
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
      ],
    },
    {
      model_id: "onnx-internal-testing/tiny-random-unispeech-ONNX",
      dtype: "fp32",
      architectures: ["UniSpeechForSequenceClassification"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "Conv",
        "Div",
        "Erf",
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
      ],
    },
  ],
};
