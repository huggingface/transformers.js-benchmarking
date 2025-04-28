// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "isnet",
  models: [
    {
      model_id: "onnx-community/ISNet-ONNX",
      dtype: "quantized",
      architectures: [],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "ConvInteger",
        "DynamicQuantizeLinear",
        "MaxPool",
        "Mul",
        "Relu",
        "Resize",
        "Sigmoid",
      ],
    },
    {
      model_id: "onnx-community/ormbg-ONNX",
      dtype: "q4",
      architectures: [],
      ops: ["Add", "Concat", "Conv", "MaxPool", "Relu", "Resize", "Sigmoid"],
    },
    {
      model_id: "onnx-community/ormbg-ONNX",
      dtype: "q4f16",
      architectures: [],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Conv",
        "MaxPool",
        "Relu",
        "Resize",
        "Sigmoid",
      ],
    },
    {
      model_id: "onnx-community/ormbg-ONNX",
      dtype: "quantized",
      architectures: [],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "ConvInteger",
        "DynamicQuantizeLinear",
        "MaxPool",
        "Mul",
        "Relu",
        "Reshape",
        "Resize",
        "Sigmoid",
      ],
    },
  ],
};
