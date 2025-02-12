import { TINY_DUMMY_IMAGE } from "../inputs.js";
import { toBeCloseToNested } from "../utils.js";

export default {
  name: "Image to Image",
  config: {
    task: "image-to-image",
    model_id: "hf-internal-testing/tiny-random-Swin2SRForImageSuperResolution",
  },
  tests: [
    {
      name: "Default",
      inputs: [TINY_DUMMY_IMAGE],
      expected: {
        size: [64, 32],
        channels: 3,
        mean: 110.57991536458333,
      },
      test_function: (result, expected) =>
        toBeCloseToNested(
          {
            size: result.size,
            channels: result.channels,
            mean: result.data.reduce((a, b) => a + b, 0) / result.data.length,
          },
          expected,
        ),
    },
  ],
};
