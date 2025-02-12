import { DUMMY_IMAGE } from "../inputs.js";
import { pick, toBeCloseToNested } from "../utils.js";

throw new Error("Not implemented");
export default {
  name: "Image Segmentation",
  config: {
    task: "image-segmentation",
    model_id: "hf-internal-testing/tiny-random-DetrForSegmentation",
  },
  tests: [
    {
      name: "Default",
      inputs: [DUMMY_IMAGE, { threshold: 0.0, mask_threshold: 0.0 }],
      expected: [
        {
          score: 0.023990020155906677,
          label: "LABEL_29",
        },
      ],
      // test_function: (result, expected) =>
      //   toBeCloseToNested(
      //     result: pick(x, ["score", "label"]),
      //     expected,
      //   ),
    },
  ],
};
