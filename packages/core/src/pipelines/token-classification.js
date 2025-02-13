const BATCH_SIZE = 256;

export default {
  name: "Token Classification",
  config: {
    task: "token-classification",
    model_id: "hf-internal-testing/tiny-random-BertForTokenClassification",
  },
  tests: [
    {
      name: "Default",
      inputs: [Array.from({ length: BATCH_SIZE }, () => "a b c d e")],
      expected: Array.from({ length: BATCH_SIZE }, () => [
        { entity: "LABEL_0", score: 0.5395538210868835, index: 1, word: "a" },
        { entity: "LABEL_0", score: 0.5303943753242493, index: 2, word: "b" },
        { entity: "LABEL_0", score: 0.5155872106552124, index: 3, word: "c" },
        { entity: "LABEL_0", score: 0.5232135653495789, index: 4, word: "d" },
        { entity: "LABEL_0", score: 0.5132490992546082, index: 5, word: "e" },
      ]),
    },
  ],
};
