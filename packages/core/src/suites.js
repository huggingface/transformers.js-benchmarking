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
      pipeline(task, model_id, options ?? DEFAULT_MODEL_OPTIONS),
    );

    const stats = {};
    for (const test of this.tests) {
      const { inputs, expected, test_function, num_runs } = test;

      const times = [];
      const numRuns = DEFAULT_NUM_WARMUP_RUNS + (num_runs ?? DEFAULT_NUM_RUNS);
      for (let i = 0; i < numRuns; ++i) {
        const { result, time: executionTime } = await time(
            () => pipe(...inputs),
        );
        const { pass, message } = (test_function ?? toBeCloseToNested)(
          result,
          expected,
        );
        if (!pass) {
          console.log(result);
          console.log(expected);
          throw new Error(
            `Test "${this.name} (${test.name})" failed: ${message()}`,
          );
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
  constructor(config) {
    this.config = config;
  }

  async *run() {
    // NOTE: Perform one test at a time to ensure accurate timing
    for await (const test of this.collect()) {
      const result = await test.run();
      yield { name: test.name, result };
    }
  }
}

/**
 * Test that the pipeline API operates correctly
 * This is the most common way of using Transformers.js
 */
export class PipelineTestSuite extends BaseTestSuite {
  constructor() {
    super(Object.values(TASKS));
  }

  *collect() {
    for (const task of this.config) {
      if (task.skip) continue;
      yield new PipelineTest(task);
    }
  }
}

export class ModelTestSuite extends BaseTestSuite {
  constructor() {
    super(MODELS);
  }

  *collect() {
    for (const [model_type, tests] of Object.entries(this.config)) {
      for (const { name, test, config } of tests) {
        yield new test(config);
      }
    }
  }
}
