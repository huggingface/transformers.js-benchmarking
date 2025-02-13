import { ModelTestSuite, PipelineTestSuite } from "@benchmarking/core";

const SUITES = {
  ModelTestSuite,
  PipelineTestSuite,
};

for (const [name, Suite] of Object.entries(SUITES)) {
  console.log("=".repeat(80));
  console.log(`Running ${name}`);

  // Run tests
  const suite = new Suite();
  for await (const { name, result } of suite.run()) {
    console.log(`  - ${name}`);
    console.log(`    - Setup time: ${result.setupTime} ms`);
    console.log(`    - Dispose time: ${result.disposeTime} ms`);
    console.log(`    - Stats:`);
    for (const [testName, stats] of Object.entries(result.stats)) {
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
  console.log("=".repeat(80));
}
