import {
  AutoProcessor,
  WhisperForConditionalGeneration,
} from "@huggingface/transformers";

import { DEFAULT_NUM_RUNS, DEFAULT_NUM_WARMUP_RUNS, Test } from "../test.js";

import { DUMMY_AUDIO } from "../inputs.js";
import { computeStatistics, time, toBeCloseToNested } from "../utils.js";

class WhisperTest extends Test {
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
      const model_id = "onnx-community/whisper-tiny.en";
      const processor = await AutoProcessor.from_pretrained(model_id);
      const model = await WhisperForConditionalGeneration.from_pretrained(
        model_id,
        {
          ...this.options,
        },
      );
      const audio = DUMMY_AUDIO;
      const inputs = await processor(audio);

      const expected = [
        "<|startoftranscript|><|notimestamps|> you<|endoftext|>",
      ];
      return [
        async () => {
          const outputs = await model.generate({
            ...inputs,
            max_new_tokens: 5,
          });
          const decoded = processor.batch_decode(outputs);
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
    test: WhisperTest,
    config: {
      name: "WhisperForConditionalGeneration",
      num_runs: 5,
    },
  },
];
