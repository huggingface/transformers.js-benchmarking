import {
  AutoTokenizer,
  AutoImageProcessor,
  CLIPModel,
} from "@huggingface/transformers";

import { DEFAULT_NUM_RUNS, DEFAULT_NUM_WARMUP_RUNS, Test } from "../test.js";

import { DUMMY_IMAGE } from "../inputs.js";
import { computeStatistics, time, toBeCloseToNested } from "../utils.js";
import { DEFAULT_MODEL_OPTIONS } from "../defaults.js";

class CLIPTest extends Test {
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
      const model_id = "onnx-internal-testing/tiny-random-CLIPModel-ONNX";

      const tokenizer = await AutoTokenizer.from_pretrained(model_id);
      const processor = await AutoImageProcessor.from_pretrained(model_id);
      const model = await CLIPModel.from_pretrained(model_id, {
        ...DEFAULT_MODEL_OPTIONS,
        ...this.options,
      });
      const texts = ["a photo of a car", "a photo of a football match"];
      const text_inputs = tokenizer(texts, { padding: true, truncation: true });
      const image_inputs = await processor(DUMMY_IMAGE);

      const expected = {
        logits_per_image: [1, 2],
        logits_per_text: [2, 1],
        text_embeds: [2, 64],
        image_embeds: [1, 64],
      };
      return [
        async () => {
          const {
            logits_per_image,
            logits_per_text,
            text_embeds,
            image_embeds,
          } = await model({
            ...text_inputs,
            ...image_inputs,
          });
          return {
            logits_per_image: logits_per_image.dims,
            logits_per_text: logits_per_text.dims,
            text_embeds: text_embeds.dims,
            image_embeds: image_embeds.dims,
          };
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
    test: CLIPTest,
    config: {
      name: "CLIPModel",
      num_runs: 20,
    },
  },
];
