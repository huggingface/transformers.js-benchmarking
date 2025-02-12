import { DUMMY_IMAGE } from "../inputs.js";

export default {
  name: "Image Classification",
  config: {
    task: "image-classification",
    model_id: "hf-internal-testing/tiny-random-vit",
  },
  tests: [
    {
      name: "Default",
      inputs: [DUMMY_IMAGE, { top_k: 3 }],
      expected: [
        { score: 0.501950204372406, label: "LABEL_0" },
        { score: 0.4980497658252716, label: "LABEL_1" },
      ],
    },
  ],
};
