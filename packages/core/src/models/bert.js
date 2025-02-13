import { AutoTokenizer, BertModel } from "@huggingface/transformers";

import { DEFAULT_NUM_RUNS, DEFAULT_NUM_WARMUP_RUNS, Test } from "../test.js";

import { computeStatistics, pick, time, toBeCloseToNested } from "../utils.js";

class BertTest extends Test {
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
      const model_id = "Snowflake/snowflake-arctic-embed-xs";
      const tokenizer = await AutoTokenizer.from_pretrained(model_id);
      const model = await BertModel.from_pretrained(model_id, {
        ...this.options,
      });
      const inputs = await tokenizer(["hello", "hello world"], {
        truncation: true,
        padding: true,
      });

      return [
        async () => {
          const { last_hidden_state } = await model(inputs);
          return pick(last_hidden_state, ["type", "dims"]);
        },
        {
          type: "float32",
          dims: [2, 4, 384],
        },
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
    test: BertTest,
    config: {
      name: "BertModel",
      num_runs: 20,
    },
  },
];
