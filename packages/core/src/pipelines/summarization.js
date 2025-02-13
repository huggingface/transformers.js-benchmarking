export default {
  name: "Summarization",
  config: {
    task: "summarization",
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
      expected: [{ summary_text: "" }],
    },
  ],
};
