export default {
  name: "Text Generation",
  config: {
    task: "text-generation",
    model_id: "hf-internal-testing/tiny-random-LlamaForCausalLM",
  },
  tests: [
    {
      name: "Default",
      inputs: [
        "a ".repeat(256),
        {
          max_new_tokens: 10,
          return_full_text: false,
        },
      ],
      expected: [
        {
          generated_text:
            " note GET unixkunftassoc joskyabestanden________leitung",
        },
      ],
    },
  ],
};
