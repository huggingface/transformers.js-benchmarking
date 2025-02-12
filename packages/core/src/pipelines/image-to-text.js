import { DUMMY_IMAGE } from "../inputs.js";

export default {
  name: "Image to Text",
  config: {
    task: "image-to-text",
    model_id:
      "hf-internal-testing/tiny-random-VisionEncoderDecoderModel-vit-gpt2",
  },
  tests: [
    {
      name: "Default",
      inputs: [DUMMY_IMAGE, { max_new_tokens: 5 }],
      expected: [{ generated_text: "ririririri" }],
    },
  ],
};
