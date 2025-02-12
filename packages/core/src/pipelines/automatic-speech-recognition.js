import { DUMMY_AUDIO } from "../inputs.js";

export default {
  name: "Automatic Speech Recognition",
  config: {
    task: "automatic-speech-recognition",
    model_id:
      "hf-internal-testing/tiny-random-MoonshineForConditionalGeneration",
  },
  tests: [
    {
      name: "Default",
      inputs: [DUMMY_AUDIO, { max_new_tokens: 3, language: "en" }],
      expected: { text: "operator Swedishapprox" },
    },
  ],
};
