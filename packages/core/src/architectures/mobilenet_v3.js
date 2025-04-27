// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "mobilenet_v3",
  models: [
    {
      model_id: "onnx-community/mobilenetv3_large_100.miil_in21k_ft_in1k",
      dtype: "q4",
      architectures: [],
      ops: [
        "Add",
        "Conv",
        "Flatten",
        "Gemm",
        "GlobalAveragePool",
        "HardSigmoid",
        "Mul",
        "ReduceMean",
        "Relu",
      ],
    },
    {
      model_id: "onnx-community/mobilenetv3_large_100.miil_in21k_ft_in1k",
      dtype: "quantized",
      architectures: [],
      ops: [
        "Add",
        "Cast",
        "ConvInteger",
        "DynamicQuantizeLinear",
        "Flatten",
        "GlobalAveragePool",
        "HardSigmoid",
        "MatMulInteger",
        "Mul",
        "ReduceMean",
        "Relu",
        "Reshape",
      ],
    },
    {
      model_id: "onnx-community/mobilenetv3_large_100.miil_in21k_ft_in1k",
      dtype: "fp16",
      architectures: [],
      ops: [
        "Add",
        "Cast",
        "Conv",
        "Flatten",
        "Gemm",
        "GlobalAveragePool",
        "HardSigmoid",
        "Mul",
        "ReduceMean",
        "Relu",
      ],
    },
  ],
};
