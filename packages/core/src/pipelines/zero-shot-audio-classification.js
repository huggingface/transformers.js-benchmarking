import { DUMMY_AUDIO } from "../inputs.js";

export default {
  name: "Zero-shot Audio Classification",
  config: {
    task: "zero-shot-audio-classification",
    model_id: "hf-internal-testing/tiny-clap-htsat-unfused",
  },
  tests: [
    {
      name: "Default",
      inputs: [
        DUMMY_AUDIO, // audio
        ["cat", "dog"], // labels
        {
          hypothesis_template: "sound of a {}", // hypothesis template
        },
      ],
      expected: [
        { score: 0.49881038069725037, label: "cat" },
        { score: 0.501189649105072, label: "dog" },
      ],
    },
  ],
};
