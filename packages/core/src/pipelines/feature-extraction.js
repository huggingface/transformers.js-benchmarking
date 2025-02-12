import { pick, toBeCloseToNested } from "../utils.js";

export default {
  name: "Feature Extraction",
  config: {
    task: "feature-extraction",
    model_id: "hf-internal-testing/tiny-random-BertModel",
  },
  tests: [
    {
      name: "Default",
      inputs: [Array.from({ length: 64 }, () => "This is a simple test.")],
      expected: {
        type: "float32",
        dims: [64, 20, 32],
        mean: -1.538501215314625e-9,
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
