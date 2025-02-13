import { toBeCloseToNested } from "../utils.js";

export default {
  name: "Text to Audio",
  config: {
    task: "text-to-audio",
    model_id: "Xenova/tiny-random-vits",
  },
  tests: [
    {
      name: "Default",
      inputs: ["hello ".repeat(32)],
      expected: {
        audio_length: 212480,
        mean: -0.0125,
      },
      test_function: (result, expected) =>
        toBeCloseToNested(
          {
            audio_length: result.audio.length,
            mean: result.audio.reduce((a, b) => a + b, 0) / result.audio.length,
          },
          expected,
          2, // The mean value is not deterministic, so we just check the first few digits
        ),
    },
  ],
};
