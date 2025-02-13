import { DUMMY_IMAGE } from "../inputs.js";

export default {
  name: "Image Segmentation",
  config: {
    task: "image-segmentation",
    model_id: "...", // TODO: Find a tiny model ID
  },
  tests: [
    {
      name: "Default",
      inputs: [DUMMY_IMAGE, { threshold: 0.0, mask_threshold: 0.0 }],
      expected: [],
    },
  ],
};
