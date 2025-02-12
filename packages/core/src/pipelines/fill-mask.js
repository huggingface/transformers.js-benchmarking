const BATCH_SIZE = 32;

export default {
  name: "Fill Mask",
  config: {
    task: "fill-mask",
    model_id: "hf-internal-testing/tiny-random-BertForMaskedLM",
  },
  tests: [
    {
      name: "Default",
      inputs: [
        Array.from(
          { length: BATCH_SIZE },
          () => "The quick brown [MASK] jumps over the lazy dog.",
        ),
        {
          top_k: 3,
        },
      ],
      expected: Array.from({ length: BATCH_SIZE }, () => [
        {
          score: 0.001336989109404385,
          token: 823,
          token_str: "##ن",
          sequence: "the quick brownن jumps over the lazy dog.",
        },
        {
          score: 0.0013063998194411397,
          token: 962,
          token_str: "##ち",
          sequence: "the quick brownち jumps over the lazy dog.",
        },
        {
          score: 0.0012906234478577971,
          token: 854,
          token_str: "##ο",
          sequence: "the quick brownο jumps over the lazy dog.",
        },
      ]),
    },
  ],
};
