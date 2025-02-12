import { DUMMY_IMAGE } from "../inputs.js";
import { pick, toBeCloseToNested } from "../utils.js";

export default {
  name: "Image Feature Extraction",
  config: {
    task: "image-feature-extraction",
    model_id: "hf-internal-testing/tiny-random-ViTMAEModel",
  },
  tests: [
    {
      name: "Default",
      inputs: [DUMMY_IMAGE],
      expected: {
        type: "float32",
        dims: [1, 91, 32],
        mean: -8.507473614471905e-10,
      },
      test_function: (result, expected) =>
        toBeCloseToNested(
          {
            ...pick(result, ["type", "dims"]),
            mean: result.mean().item(),
          },
          expected,
        ),
    },
  ],
};
