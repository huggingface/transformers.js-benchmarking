// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "rt_detr",
  models: [
    {
      model_id: "onnx-community/rtdetr_r18vd_coco_o365",
      dtype: "q4",
      architectures: ["RTDetrForObjectDetection"],
      ops: [
        "Add",
        "AveragePool",
        "Clip",
        "Concat",
        "Conv",
        "Div",
        "Erf",
        "Gather",
        "GatherElements",
        "GridSample",
        "Log",
        "MatMul",
        "MatMulNBits",
        "MaxPool",
        "Mul",
        "Pow",
        "ReduceMax",
        "ReduceMean",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Resize",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
      ],
    },
    {
      model_id: "onnx-community/rtdetr_r18vd_coco_o365",
      dtype: "bnb4",
      architectures: ["RTDetrForObjectDetection"],
      ops: [
        "Add",
        "AveragePool",
        "Clip",
        "Concat",
        "Conv",
        "Div",
        "Erf",
        "Gather",
        "GatherElements",
        "GridSample",
        "Log",
        "MatMul",
        "MatMulBnb4",
        "MaxPool",
        "Mul",
        "Pow",
        "ReduceMax",
        "ReduceMean",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Resize",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
      ],
    },
    {
      model_id: "onnx-community/rtdetr_r18vd_coco_o365",
      dtype: "quantized",
      architectures: ["RTDetrForObjectDetection"],
      ops: [
        "Add",
        "AveragePool",
        "Cast",
        "Clip",
        "Concat",
        "ConvInteger",
        "Div",
        "DynamicQuantizeLinear",
        "Erf",
        "Gather",
        "GatherElements",
        "GridSample",
        "Log",
        "MatMul",
        "MatMulInteger",
        "MaxPool",
        "Mul",
        "Pow",
        "ReduceMax",
        "ReduceMean",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Resize",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
      ],
    },
    {
      model_id: "onnx-community/rtdetr_r18vd_coco_o365",
      dtype: "fp32",
      architectures: ["RTDetrForObjectDetection"],
      ops: [
        "Add",
        "AveragePool",
        "Clip",
        "Concat",
        "Conv",
        "Div",
        "Erf",
        "Gather",
        "GatherElements",
        "GridSample",
        "Log",
        "MatMul",
        "MaxPool",
        "Mul",
        "Pow",
        "ReduceMax",
        "ReduceMean",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Resize",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
      ],
    },
    {
      model_id: "onnx-community/rtdetr_r18vd_coco_o365",
      dtype: "fp16",
      architectures: ["RTDetrForObjectDetection"],
      ops: [
        "Add",
        "AveragePool",
        "Cast",
        "Clip",
        "Concat",
        "Conv",
        "Div",
        "Erf",
        "Gather",
        "GatherElements",
        "GridSample",
        "Log",
        "MatMul",
        "MaxPool",
        "Mul",
        "Pow",
        "ReduceMax",
        "ReduceMean",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Resize",
        "Shape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Sub",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
      ],
    },
  ],
};
