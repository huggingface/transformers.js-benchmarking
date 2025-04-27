// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "hifigan",
  models: [
    {
      model_id: "Xenova/speecht5_hifigan",
      dtype: "quantized",
      architectures: ["SpeechT5HifiGan"],
      ops: [
        "Add",
        "Cast",
        "Constant",
        "ConvInteger",
        "ConvTranspose",
        "Div",
        "DynamicQuantizeLinear",
        "LeakyRelu",
        "Mul",
        "Reshape",
        "Squeeze",
        "Sub",
        "Tanh",
        "Transpose",
        "Unsqueeze",
      ],
    },
    {
      model_id: "Xenova/speecht5_hifigan",
      dtype: "fp32",
      architectures: ["SpeechT5HifiGan"],
      ops: [
        "Add",
        "Constant",
        "Conv",
        "ConvTranspose",
        "Div",
        "LeakyRelu",
        "Reshape",
        "Squeeze",
        "Sub",
        "Tanh",
        "Transpose",
        "Unsqueeze",
      ],
    },
  ],
};
