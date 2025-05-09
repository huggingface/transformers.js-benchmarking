// NOTE: This file has been auto-generated. Do not edit directly.

export default {
  model_type: "rf_detr",
  models: [
    {
      model_id: "onnx-community/rfdetr_base-ONNX",
      dtype: "q4",
      architectures: [],
      ops: [
        "Add",
        "And",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "Conv",
        "Cos",
        "Div",
        "Equal",
        "Erf",
        "Exp",
        "Expand",
        "Gather",
        "GatherElements",
        "Greater",
        "GridSample",
        "LayerNormalization",
        "Less",
        "MatMul",
        "MatMulNBits",
        "Mul",
        "Not",
        "Pow",
        "Range",
        "ReduceMax",
        "ReduceMean",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Resize",
        "ScatterND",
        "Shape",
        "Sigmoid",
        "Sin",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/rfdetr_base-ONNX",
      dtype: "bnb4",
      architectures: [],
      ops: [
        "Add",
        "And",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "Conv",
        "Cos",
        "Div",
        "Equal",
        "Erf",
        "Exp",
        "Expand",
        "Gather",
        "GatherElements",
        "Greater",
        "GridSample",
        "LayerNormalization",
        "Less",
        "MatMul",
        "MatMulBnb4",
        "Mul",
        "Not",
        "Pow",
        "Range",
        "ReduceMax",
        "ReduceMean",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Resize",
        "ScatterND",
        "Shape",
        "Sigmoid",
        "Sin",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/rfdetr_base-ONNX",
      dtype: "quantized",
      architectures: [],
      ops: [
        "Add",
        "And",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "ConvInteger",
        "Cos",
        "Div",
        "DynamicQuantizeLinear",
        "Equal",
        "Erf",
        "Exp",
        "Expand",
        "Gather",
        "GatherElements",
        "Greater",
        "GridSample",
        "LayerNormalization",
        "Less",
        "MatMul",
        "MatMulInteger",
        "Mul",
        "Not",
        "Pow",
        "Range",
        "ReduceMax",
        "ReduceMean",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Resize",
        "ScatterND",
        "Shape",
        "Sigmoid",
        "Sin",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/rfdetr_base-ONNX",
      dtype: "fp32",
      architectures: [],
      ops: [
        "Add",
        "And",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "Conv",
        "Cos",
        "Div",
        "Equal",
        "Erf",
        "Exp",
        "Expand",
        "Gather",
        "GatherElements",
        "Greater",
        "GridSample",
        "LayerNormalization",
        "Less",
        "MatMul",
        "Mul",
        "Not",
        "Pow",
        "Range",
        "ReduceMax",
        "ReduceMean",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Resize",
        "ScatterND",
        "Shape",
        "Sigmoid",
        "Sin",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/rfdetr_large-ONNX",
      dtype: "q4",
      architectures: [],
      ops: [
        "Add",
        "And",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "Conv",
        "ConvTranspose",
        "Cos",
        "Div",
        "Equal",
        "Erf",
        "Exp",
        "Expand",
        "Gather",
        "GatherElements",
        "Greater",
        "GridSample",
        "LayerNormalization",
        "Less",
        "MatMul",
        "MatMulNBits",
        "Mul",
        "Not",
        "Pow",
        "Range",
        "ReduceMax",
        "ReduceMean",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Resize",
        "ScatterND",
        "Shape",
        "Sigmoid",
        "Sin",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/rfdetr_large-ONNX",
      dtype: "bnb4",
      architectures: [],
      ops: [
        "Add",
        "And",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "Conv",
        "ConvTranspose",
        "Cos",
        "Div",
        "Equal",
        "Erf",
        "Exp",
        "Expand",
        "Gather",
        "GatherElements",
        "Greater",
        "GridSample",
        "LayerNormalization",
        "Less",
        "MatMul",
        "MatMulBnb4",
        "Mul",
        "Not",
        "Pow",
        "Range",
        "ReduceMax",
        "ReduceMean",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Resize",
        "ScatterND",
        "Shape",
        "Sigmoid",
        "Sin",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/rfdetr_large-ONNX",
      dtype: "quantized",
      architectures: [],
      ops: [
        "Add",
        "And",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "ConvInteger",
        "ConvTranspose",
        "Cos",
        "Div",
        "DynamicQuantizeLinear",
        "Equal",
        "Erf",
        "Exp",
        "Expand",
        "Gather",
        "GatherElements",
        "Greater",
        "GridSample",
        "LayerNormalization",
        "Less",
        "MatMul",
        "MatMulInteger",
        "Mul",
        "Not",
        "Pow",
        "Range",
        "ReduceMax",
        "ReduceMean",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Resize",
        "ScatterND",
        "Shape",
        "Sigmoid",
        "Sin",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
    {
      model_id: "onnx-community/rfdetr_large-ONNX",
      dtype: "fp32",
      architectures: [],
      ops: [
        "Add",
        "And",
        "Cast",
        "Concat",
        "ConstantOfShape",
        "Conv",
        "ConvTranspose",
        "Cos",
        "Div",
        "Equal",
        "Erf",
        "Exp",
        "Expand",
        "Gather",
        "GatherElements",
        "Greater",
        "GridSample",
        "LayerNormalization",
        "Less",
        "MatMul",
        "Mul",
        "Not",
        "Pow",
        "Range",
        "ReduceMax",
        "ReduceMean",
        "ReduceSum",
        "Relu",
        "Reshape",
        "Resize",
        "ScatterND",
        "Shape",
        "Sigmoid",
        "Sin",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
        "Where",
      ],
    },
  ],
};
