import { DUMMY_IMAGE } from "../inputs.js";

export default {
  name: "Object Detection",
  config: {
    task: "object-detection",
    model_id: "onnx-internal-testing/tiny-random-YolosForObjectDetection-ONNX",
  },
  tests: [
    {
      name: "Default",
      inputs: [DUMMY_IMAGE, { threshold: 0 }],
      expected: [],
    },
  ],
};
