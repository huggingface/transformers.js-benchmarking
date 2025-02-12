import { DUMMY_AUDIO } from "../inputs.js";

export default {
  name: "Audio Classification",
  config: {
    task: "audio-classification",
    model_id: "hf-internal-testing/tiny-random-unispeech",
  },
  tests: [
    {
      name: "Default",
      inputs: [DUMMY_AUDIO, { top_k: 3 }],
      expected: [
        { score: 0.5043687224388123, label: "LABEL_0" },
        { score: 0.4956313371658325, label: "LABEL_1" },
      ],
    },
  ],
};
