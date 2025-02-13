import { SamModel, Tensor } from "@huggingface/transformers";

import { DEFAULT_NUM_RUNS, DEFAULT_NUM_WARMUP_RUNS, Test } from "../test.js";

import { computeStatistics, pick, time, toBeCloseToNested } from "../utils.js";

class SamTest extends Test {
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
      const model_id = "Xenova/slimsam-77-uniform";
      const model = await SamModel.from_pretrained(model_id, {
        ...this.options,
      });

      const inputs = {
        input_points: new Tensor("float32", [10, 10], [1, 1, 1, 2]),
        input_labels: new Tensor("int64", [0], [1, 1, 1]),
        image_embeddings: new Tensor(
          "float32",
          new Float32Array(1 * 256 * 64 * 64),
          [1, 256, 64, 64],
        ),
        image_positional_embeddings: new Tensor(
          "float32",
          new Float32Array(1 * 256 * 64 * 64),
          [1, 256, 64, 64],
        ),
      };

      return [
        async () => {
          const { pred_masks, iou_scores } = await model(inputs);
          return {
            pred_masks: pick(pred_masks, ["type", "dims"]),
            iou_scores: pick(iou_scores, ["type", "dims"]),
          };
        },
        {
          pred_masks: {
            type: "float32",
            dims: [1, 1, 3, 256, 256],
          },
          iou_scores: {
            type: "float32",
            dims: [1, 1, 3],
          },
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
    test: SamTest,
    config: {
      name: "SamModel",
      num_runs: 20,
    },
  },
];
