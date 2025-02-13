import { ModelTestSuite, PipelineTestSuite } from "@benchmarking/core";

const SUITES = {
  ModelTestSuite,
  PipelineTestSuite,
};

for (const [suiteName, Suite] of Object.entries(SUITES)) {
  const FAILURES = [];
  console.log("=".repeat(80));
  console.log(`Running ${suiteName}`);

  // Run tests
  const suite = new Suite();
  for await (const { name, result } of suite.run()) {
    console.log(`  - ${name}`);
    if ("error" in result) {
      console.error(`    - Error: ${result.error}`);
      FAILURES.push({ suite: suiteName, name, error: result.error });
      continue;
    }
    console.log(`    - Setup time: ${result.setupTime} ms`);
    console.log(`    - Dispose time: ${result.disposeTime} ms`);
    console.log(`    - Stats:`);
    for (const [testName, stats] of Object.entries(
      result.stats as Record<string, any>,
    )) {
      console.log(`      - ${testName}`);
      console.log(`        - Mean: ${stats.mean} ms`);
      console.log(`        - Median: ${stats.median} ms`);
      console.log(`        - Min: ${stats.min} ms`);
      console.log(`        - Max: ${stats.max} ms`);
      console.log(
        `        - Percentiles: 1st=${stats.p1} ms, 5th=${stats.p5} ms, 10th=${stats.p10} ms, 90th=${stats.p90} ms, 95th=${stats.p95} ms, 99th=${stats.p99} ms`,
      );
      console.log(`        - Standard deviation: ${stats.stdDev} ms`);
    }
  }

  if (FAILURES.length > 0) {
    console.table(FAILURES);
  }
  console.log("=".repeat(80));
}
