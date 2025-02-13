export default {
  name: "Text to Text Generation",
  config: {
    task: "text2text-generation",
    model_id: "hf-internal-testing/tiny-random-T5ForConditionalGeneration",
  },
  tests: [
    {
      name: "Default",
      inputs: [
        "a ".repeat(256),
        {
          max_new_tokens: 10,
        },
      ],
      expected: [{ generated_text: "" }],
    },
  ],
};
