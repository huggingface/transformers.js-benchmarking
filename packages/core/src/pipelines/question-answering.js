export default {
  name: "Question Answering",
  config: {
    task: "question-answering",
    model_id: "hf-internal-testing/tiny-random-BertForQuestionAnswering",
  },
  tests: [
    {
      name: "Default",
      inputs: [
        "a ".repeat(256),
        "b c ".repeat(256),
        {
          top_k: 1,
        },
      ],
      expected: {
        score: 0.000020034284354665714,
        /* start: 0, end: 1, */ answer:
          "c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b c b",
      },
    },
  ],
};
