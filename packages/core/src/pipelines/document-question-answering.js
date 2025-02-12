import { TINY_DUMMY_IMAGE } from "../inputs.js";

export default {
  name: "Document Question Answering",
  config: {
    task: "document-question-answering",
    model_id:
      "hf-internal-testing/tiny-random-VisionEncoderDecoderModel-vit-gpt2",
  },
  tests: [
    {
      name: "Default",
      inputs: [
        TINY_DUMMY_IMAGE,
        "What is the invoice number?",
        { max_new_tokens: 10 },
      ],
      expected: [{ answer: null }],
    },
  ],
};
