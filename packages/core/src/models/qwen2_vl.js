
import {
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
  } from "@huggingface/transformers";
  
import { DEFAULT_MODEL_OPTIONS } from "../defaults.js";
import { DEFAULT_NUM_RUNS, DEFAULT_NUM_WARMUP_RUNS, Test } from "../test.js";

import { DUMMY_IMAGE } from "../inputs.js";
import { computeStatistics, time, toBeCloseToNested } from "../utils.js";

class Qwen2VLTest extends Test {
  constructor(config) {
    super(config);
    this.name = config.name;
    this.num_runs = config.num_runs ?? DEFAULT_NUM_RUNS;
  }
  async run() {
    const {
      result: [test, expected, cleanup],
      time: setupTime,
    } = await time(async () => {
      const model_id =
        "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration";
      const processor = await AutoProcessor.from_pretrained(model_id);
      const model = await Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        DEFAULT_MODEL_OPTIONS,
      );
      const image = DUMMY_IMAGE;
      const conversation = [
        {
          role: "user",
          content: [
            { type: "image" },
            { type: "text", text: "Describe this image." },
          ],
        },
      ];
      const text = processor.apply_chat_template(conversation, {
        add_generation_prompt: true,
      });
      const inputs = await processor(text, image);

      const expected = [" finishing Patio无意 możliwości𬱖"];
      return [
        async () => {
          const outputs = await model.generate({
            ...inputs,
            max_new_tokens: 5,
          });
          const decoded = processor.batch_decode(
            outputs.slice(null, [inputs.input_ids.dims.at(-1), null]),
            { skip_special_tokens: true },
          );
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
        throw new Error(`Test "${this.name}" failed: ${message()}`);
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
    test: Qwen2VLTest,
    config: {
      name: "Qwen2VL",
      num_runs: 10,
    },
  },
];
