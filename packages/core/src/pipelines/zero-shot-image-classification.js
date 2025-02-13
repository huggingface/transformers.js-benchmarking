import { DUMMY_IMAGE } from "../inputs.js";

export default {
  name: "Zero-shot Image Classification",
  config: {
    task: "zero-shot-image-classification",
    model_id: "hf-internal-testing/tiny-random-GroupViTModel",
  },
  tests: [
    {
      name: "Default",
      inputs: [
        DUMMY_IMAGE, // image
        ["cat", "dog"], // labels
        {
          hypothesis_template: "a photo of a {}", // hypothesis template
        },
      ],
      expected: [
        { score: 0.5883981585502625, label: "dog" },
        { score: 0.41160184144973755, label: "cat" },
      ],
    },
  ],
};
