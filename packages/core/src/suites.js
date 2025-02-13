import { pipeline } from "@huggingface/transformers";

import { computeStatistics, toBeCloseToNested, time } from "./utils.js";
import { DEFAULT_MODEL_OPTIONS } from "./defaults.js";

import { DEFAULT_NUM_RUNS, DEFAULT_NUM_WARMUP_RUNS, Test } from "./test.js";

import * as TASKS from "./tasks.js";
import * as MODELS from "./models.js";

class PipelineTest extends Test {
  constructor(data) {
    super(data);
    this.name = data.name;
    this.config = data.config;
    this.tests = data.tests;
  }

  async run() {
    const { task, model_id, options } = this.config;

    const { result: pipe, time: setupTime } = await time(() =>
      pipeline(task, model_id, {
        ...DEFAULT_MODEL_OPTIONS,
        ...options,
      }),
    );

    const stats = {};
    for (const test of this.tests) {
      const { inputs, expected, test_function, num_runs } = test;

      const times = [];
      const numRuns = DEFAULT_NUM_WARMUP_RUNS + (num_runs ?? DEFAULT_NUM_RUNS);
      for (let i = 0; i < numRuns; ++i) {
        const { result, time: executionTime } = await time(() =>
          pipe(...inputs),
        );
        const { pass, message } = (test_function ?? toBeCloseToNested)(
          result,
          expected,
        );
        if (!pass) {
          console.log(result);
          console.log(expected);
          throw new Error(message());
        }
        if (i >= DEFAULT_NUM_WARMUP_RUNS) times.push(executionTime);
      }
      stats[test.name] = computeStatistics(times);
    }

    const { time: disposeTime } = await time(() => pipe.dispose());

    return {
      setupTime,
      stats,
      disposeTime,
    };
  }
}

class BaseTestSuite {
  constructor(config, options) {
    this.config = config;
    this.options = options;
  }

  async *run() {
    // NOTE: Perform one test at a time to ensure accurate timing
    for await (const test of this.collect()) {
      const result = await test.run().catch((error) => {
        console.error(error);
        return { error: error.message };
      });
      yield { name: test.name, result };
    }
  }
}

/**
 * Test that the pipeline API operates correctly
 * This is the most common way of using Transformers.js
 */
export class PipelineTestSuite extends BaseTestSuite {
  constructor(options) {
    super(Object.values(TASKS), options);
  }

  *collect() {
    for (const task of this.config) {
      if (task.skip) continue;
      yield new PipelineTest({
        ...task,
        config: { ...task.config, options: this.options },
      });
    }
  }
}

export class ModelTestSuite extends BaseTestSuite {
  constructor(options) {
    super(MODELS, options);
  }

  *collect() {
    for (const [model_type, tests] of Object.entries(this.config)) {
      for (const { name, test, config } of tests) {
        yield new test({
          ...config,
          options: this.options,
        });
      }
    }
  }
}
