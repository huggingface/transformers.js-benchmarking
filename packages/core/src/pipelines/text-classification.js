export default {
  name: "Text Classification",
  config: {
    task: "text-classification",
    model_id: "hf-internal-testing/tiny-random-BertForSequenceClassification",
  },
  tests: [
    {
      name: "Default",
      inputs: [
        "a ".repeat(512),
        {
          top_k: 1,
        },
      ],
      expected: [{ label: "LABEL_0", score: 0.5076301693916321 }],
    },
  ],
};
