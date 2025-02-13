import { useState, useRef, useEffect } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

export default function TestSuiteUI() {
  const [results, setResults] = useState({});
  const [isRunning, setIsRunning] = useState(false);
  const [selectedSuite, setSelectedSuite] = useState("PipelineTestSuite");
  const [selectedDevice, setSelectedDevice] = useState("wasm");
  const worker = useRef(null);
  const testRefs = useRef({});

  useEffect(() => {
    if (worker.current) return;
    worker.current = new Worker(new URL("./worker.js", import.meta.url), {
      type: "module",
    });
    worker.current.onmessage = (event) => {
      const data = event.data;
      if (data.status === "update") {
        const { name, result } = data;
        setResults((prev) => ({ ...prev, [name]: result }));
      } else if (data.status === "complete") {
        setIsRunning(false);
      }
    };
    return () => {
      worker.current.terminate();
      worker.current = null;
    };
  }, []);

  const scrollToTest = (testName) => {
    testRefs.current[testName]?.scrollIntoView({ behavior: "smooth" });
  };

  const toggleExpand = (testName) => {
    setExpandedTests((prev) => ({ ...prev, [testName]: !prev[testName] }));
  };

  const [expandedTests, setExpandedTests] = useState({});

  // Update expandedTests when new results come in.
  useEffect(() => {
    setExpandedTests((prev) => {
      const newState = { ...prev };
      Object.keys(results).forEach((key) => {
        if (newState[key] === undefined) {
          newState[key] = true;
        }
      });
      return newState;
    });
  }, [results]);

  const runBenchmark = () => {
    // Clear previous results and disable settings.
    setResults({});
    setIsRunning(true);
    // Send the start command to the worker with selected settings.
    worker.current.postMessage({
      command: "start",
      suite: selectedSuite,
      device: selectedDevice,
    });
  };

  return (
    <div className="bg-gradient-to-br from-gray-100 to-gray-200 min-h-screen">
      <div className="container mx-auto p-6">
        <h1 className="text-4xl font-bold mb-8 text-center text-gray-800">
          Transformers.js Benchmarking
        </h1>
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold mb-4 text-gray-800">
            Settings
          </h2>
          <div className="grid md:grid-cols-3 gap-4 items-end">
            <div>
              <label
                htmlFor="suite-select"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                Test Suite
              </label>
              <select
                id="suite-select"
                value={selectedSuite}
                onChange={(e) => setSelectedSuite(e.target.value)}
                disabled={isRunning}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2"
              >
                <option value="PipelineTestSuite">PipelineTestSuite</option>
                <option value="ModelTestSuite">ModelTestSuite</option>
              </select>
            </div>
            <div>
              <label
                htmlFor="device-select"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                Device
              </label>
              <select
                id="device-select"
                value={selectedDevice}
                onChange={(e) => setSelectedDevice(e.target.value)}
                disabled={isRunning}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2"
              >
                <option value="wasm">wasm</option>
                <option value="webgpu">webgpu</option>
              </select>
            </div>
            <div>
              <button
                onClick={runBenchmark}
                disabled={isRunning}
                className="w-full py-2 px-4 bg-blue-600 text-white rounded-md shadow hover:bg-blue-700 disabled:opacity-50"
              >
                {isRunning ? "Running..." : "Run Benchmark"}
              </button>
            </div>
          </div>
        </div>

        <div className="mb-8 overflow-x-auto">
          <table className="min-w-full bg-white shadow-md rounded-lg overflow-hidden">
            <thead className="bg-gray-200">
              <tr>
                <th className="px-4 py-2 text-left text-gray-700">Task</th>
                <th className="px-4 py-2 text-left text-gray-700">
                  Average Duration (ms)
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(results).map(([name, result]) => {
                if ("error" in result) {
                  return (
                    <tr
                      key={name}
                      className="hover:bg-gray-100 cursor-pointer bg-red-100"
                    >
                      <td className="px-4 py-2 border-t border-gray-300">
                        {name}
                      </td>
                      <td className="px-4 py-2 border-t border-gray-300">
                        Error: {result.error}
                      </td>
                    </tr>
                  );
                }
                console.log(result);
                const avgDuration =
                  Object.values(result.stats).reduce(
                    (sum, stat) => sum + stat.mean,
                    0,
                  ) / Object.values(result.stats).length;
                return (
                  <tr
                    key={name}
                    className="hover:bg-gray-100 cursor-pointer"
                    onClick={() => scrollToTest(name)}
                  >
                    <td className="px-4 py-2 border-t border-gray-300">
                      {name}
                    </td>
                    <td className="px-4 py-2 border-t border-gray-300">
                      {avgDuration.toFixed(2)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="space-y-6">
          {Object.entries(results).map(
            ([name, result]) =>
              !("error" in result) && (
                <div
                  key={name}
                  className="bg-white rounded-lg shadow-lg overflow-hidden transition-all duration-300 ease-in-out"
                  ref={(el) => (testRefs.current[name] = el)}
                >
                  <div
                    className="bg-blue-600 text-white p-4 cursor-pointer flex justify-between items-center"
                    onClick={() => toggleExpand(name)}
                  >
                    <h2 className="text-2xl font-semibold">{name}</h2>
                    {expandedTests[name] ? (
                      <ChevronUp size={24} />
                    ) : (
                      <ChevronDown size={24} />
                    )}
                  </div>
                  {expandedTests[name] && (
                    <div className="p-6 space-y-6">
                      <div className="grid grid-cols-2 gap-4 text-md">
                        <div className="bg-green-50 p-2 rounded">
                          <p className="font-semibold text-green-800">
                            Setup time
                          </p>
                          <p className="text-green-900">
                            {result.setupTime.toFixed(2)} ms
                          </p>
                        </div>
                        <div className="bg-red-50 p-2 rounded">
                          <p className="font-semibold text-red-800">
                            Dispose time
                          </p>
                          <p className="text-red-900">
                            {result.disposeTime.toFixed(2)} ms
                          </p>
                        </div>
                      </div>
                      <div className="bg-gray-50 p-6 rounded-lg shadow-inner">
                        <h3 className="text-2xl font-semibold mb-4 text-gray-800">
                          Stats (ms)
                        </h3>
                        {Object.entries(result.stats).map(
                          ([testName, stats]) => (
                            <div key={testName} className="mb-6 last:mb-0">
                              <h4 className="text-xl font-medium mb-3 text-gray-700">
                                {testName}
                              </h4>
                              <div className="grid grid-cols-3 sm:grid-cols-5 gap-4">
                                {["mean", "median", "min", "max", "stdDev"].map(
                                  (stat) => (
                                    <div
                                      key={stat}
                                      className="bg-white p-3 rounded-lg shadow"
                                    >
                                      <p className="text-sm font-medium text-gray-600 capitalize">
                                        {stat}
                                      </p>
                                      <p className="text-lg font-bold text-gray-900">
                                        {stats[stat].toFixed(2)}
                                      </p>
                                    </div>
                                  ),
                                )}
                              </div>
                            </div>
                          ),
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ),
          )}
        </div>
      </div>
    </div>
  );
}
