// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "textnet",
  models: [
    {
      model_id: "onnx-community/textnet-tiny",
      dtype: "q4",
      architectures: ["TextNetBackbone"],
      ops: ["Add", "BatchNormalization", "Conv", "Relu"],
    },
    {
      model_id: "onnx-community/textnet-tiny",
      dtype: "q4f16",
      architectures: ["TextNetBackbone"],
      ops: ["Add", "BatchNormalization", "Cast", "Conv", "Relu"],
    },
    {
      model_id: "onnx-community/textnet-tiny",
      dtype: "quantized",
      architectures: ["TextNetBackbone"],
      ops: [
        "Add",
        "BatchNormalization",
        "Cast",
        "ConvInteger",
        "DynamicQuantizeLinear",
        "Mul",
        "Relu",
        "Reshape",
      ],
    },
  ],
};
