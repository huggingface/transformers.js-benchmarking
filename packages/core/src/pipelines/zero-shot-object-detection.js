import { DUMMY_IMAGE } from "../inputs.js";

export default {
  name: "Zero-shot Object Detection",
  config: {
    task: "zero-shot-object-detection",
    model_id: "hf-internal-testing/tiny-random-OwlViTForObjectDetection",
  },
  tests: [
    {
      name: "Default",
      inputs: [
        DUMMY_IMAGE, // image
        ["cat", "dog"], // labels
        { top_k: 3 },
      ],
      expected: [
        {
          box: { xmin: 54, ymin: 44, xmax: 69, ymax: 58 },
          label: "cat",
          score: 0.6850778460502625,
        },
        {
          box: { xmin: 204, ymin: 44, xmax: 218, ymax: 59 },
          label: "cat",
          score: 0.6674455404281616,
        },
        {
          box: { xmin: 140, ymin: 70, xmax: 154, ymax: 85 },
          label: "cat",
          score: 0.6629524827003479,
        },
      ],
    },
  ],
};
