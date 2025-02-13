import {
  AutoTokenizer,
  T5ForConditionalGeneration,
} from "@huggingface/transformers";

import { DEFAULT_NUM_RUNS, DEFAULT_NUM_WARMUP_RUNS, Test } from "../test.js";
import { computeStatistics, time, toBeCloseToNested } from "../utils.js";

class T5Test extends Test {
  constructor(config) {
    super(config);
    this.name = config.name;
    this.options = config.options;
    this.num_runs = config.num_runs ?? DEFAULT_NUM_RUNS;
  }
  async run() {
    const {
      result: [test, expected, cleanup],
      time: setupTime,
    } = await time(async () => {
      const model_id = "Xenova/t5-small";
      const tokenizer = await AutoTokenizer.from_pretrained(model_id);
      const model = await T5ForConditionalGeneration.from_pretrained(model_id, {
        ...this.options,
      });
      const inputs = await tokenizer("Translate English to French: Hello.");

      const expected = ["<pad> Bonjour.</s>"];
      return [
        async () => {
          const outputs = await model.generate({
            ...inputs,
            max_new_tokens: 5,
          });
          const decoded = tokenizer.batch_decode(outputs);
          return decoded;
        },
        expected,
        () => model.dispose(),
      ];
    });

    const times = [];
    const numRuns = DEFAULT_NUM_WARMUP_RUNS + this.num_runs;
    for (let i = 0; i < numRuns; ++i) {
      const { result, time: executionTime } = await time(test);
      const { pass, message } = toBeCloseToNested(result, expected);
      if (!pass) {
        console.log(result);
        console.log(expected);
        throw new Error(message());
      }
      if (i >= DEFAULT_NUM_WARMUP_RUNS) times.push(executionTime);
    }
    const stats = {
      [this.name]: computeStatistics(times),
    };

    const { time: disposeTime } = await time(cleanup);

    return {
      setupTime,
      stats,
      disposeTime,
    };
  }
}

export default [
  {
    test: T5Test,
    config: {
      name: "T5ForConditionalGeneration",
      num_runs: 10,
    },
  },
];
