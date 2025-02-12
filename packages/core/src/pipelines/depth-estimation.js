import { DUMMY_IMAGE } from "../inputs.js";
import { pick, toBeCloseToNested } from "../utils.js";

export default {
  name: "Depth Estimation",
  config: {
    task: "depth-estimation",
    model_id: "hf-internal-testing/tiny-random-DPTForDepthEstimation",
  },
  tests: [
    {
      name: "Default",
      inputs: [DUMMY_IMAGE],
      expected: {
        predicted_depth: { dims: [224, 224] },
        depth: { size: DUMMY_IMAGE.size },
      },
      test_function: (result, expected) =>
        toBeCloseToNested(
          {
            predicted_depth: pick(result.predicted_depth, ["dims"]),
            depth: pick(result.depth, ["size"]),
          },
          expected,
        ),
    },
  ],
};
