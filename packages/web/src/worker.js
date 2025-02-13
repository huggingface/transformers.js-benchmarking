import { ModelTestSuite, PipelineTestSuite } from "@benchmarking/core";

const SUITES = {
  ModelTestSuite,
  PipelineTestSuite,
};

self.addEventListener("message", async (event) => {
  const { command, suite, device } = event.data;
  if (command !== "start") return;

  console.log(`Starting test suite: ${suite} on device: ${device}`);

  const TestSuiteClass = SUITES[suite];
  if (!TestSuiteClass) {
    console.error(`Unknown suite type: ${suite}`);
    return;
  }

  const cls = new TestSuiteClass({ device });
  for await (const { name, result } of cls.run()) {
    self.postMessage({ status: "update", name, result });
  }
  self.postMessage({ status: "complete" });
});
