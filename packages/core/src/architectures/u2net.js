// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "u2net",
  models: [
    {
      model_id: "BritishWerewolf/IS-Net",
      dtype: "fp32",
      architectures: ["U2NetModel"],
      ops: [
        "Add",
        "Concat",
        "Constant",
        "Conv",
        "MaxPool",
        "Relu",
        "Resize",
        "Shape",
        "Sigmoid",
        "Slice",
      ],
    },
    {
      model_id: "BritishWerewolf/IS-Net-Anime",
      dtype: "fp32",
      architectures: ["U2NetModel"],
      ops: ["Add", "Concat", "Conv", "MaxPool", "Relu", "Resize", "Sigmoid"],
    },
    {
      model_id: "BritishWerewolf/U-2-Net",
      dtype: "fp32",
      architectures: ["U2NetModel"],
      ops: [
        "Add",
        "Cast",
        "Concat",
        "Constant",
        "Conv",
        "Gather",
        "MaxPool",
        "Relu",
        "Resize",
        "Shape",
        "Sigmoid",
        "Slice",
        "Unsqueeze",
      ],
    },
  ],
};
