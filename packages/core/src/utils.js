/**
 * Computes the percentile value from a sorted array.
 *
 * @param {number[]} sorted A sorted array of numbers.
 * @param {number} percentile The requested percentile (0-100).
 * @returns {number} The computed percentile value.
 */
function getPercentile(sorted, percentile) {
  const n = sorted.length;
  if (n === 0) return undefined;

  const rank = (percentile / 100) * (n - 1);
  const lowerIndex = Math.floor(rank);
  const upperIndex = Math.ceil(rank);
  const weight = rank - lowerIndex;

  if (upperIndex >= n) {
    return sorted[lowerIndex];
  }
  return sorted[lowerIndex] * (1 - weight) + sorted[upperIndex] * weight;
}

/**
 * Computes statistics for an array of numbers.
 * Returns an object with properties:
 *   min, max, mean, median, p1, p5, p10, p90, p95, p99, stdDev
 *
 * @param {number[]} numbers The list of numbers.
 * @returns {Object} Statistics for the input numbers.
 */
export function computeStatistics(numbers) {
  if (!Array.isArray(numbers) || numbers.length === 0) {
    throw new Error("Input must be a non-empty array of numbers");
  }

  const n = numbers.length;
  const sum = numbers.reduce((a, b) => a + b, 0);
  const mean = sum / n;

  // Sort the array for percentile calculations
  const sorted = numbers.toSorted((a, b) => a - b);
  const min = sorted.at(0);
  const max = sorted.at(-1);

  // Calculate standard deviation (population std dev)
  const variance = numbers.reduce((acc, num) => acc + (num - mean) ** 2, 0) / n;
  const stdDev = Math.sqrt(variance);

  return {
    min,
    max,
    mean,
    median: getPercentile(sorted, 50),
    p1: getPercentile(sorted, 1),
    p5: getPercentile(sorted, 5),
    p10: getPercentile(sorted, 10),
    p90: getPercentile(sorted, 90),
    p95: getPercentile(sorted, 95),
    p99: getPercentile(sorted, 99),
    stdDev,
  };
}

export const toBeCloseToNested = (received, expected, numDigits = 5) => {
  const compare = (received, expected, path = "") => {
    if (
      typeof received === "number" &&
      typeof expected === "number" &&
      !Number.isInteger(received) &&
      !Number.isInteger(expected)
    ) {
      const pass = Math.abs(received - expected) < Math.pow(10, -numDigits);
      return {
        pass,
        message: () =>
          pass
            ? `✓ At path '${path}': expected ${received} not to be close to ${expected} with tolerance of ${numDigits} decimal places`
            : `✗ At path '${path}': expected ${received} to be close to ${expected} with tolerance of ${numDigits} decimal places`,
      };
    } else if (Array.isArray(received) && Array.isArray(expected)) {
      if (received.length !== expected.length) {
        return {
          pass: false,
          message: () =>
            `✗ At path '${path}': array lengths differ. Received length ${received.length}, expected length ${expected.length}`,
        };
      }
      for (let i = 0; i < received.length; i++) {
        const result = compare(received[i], expected[i], `${path}[${i}]`);
        if (!result.pass) return result;
      }
    } else if (
      typeof received === "object" &&
      typeof expected === "object" &&
      received !== null &&
      expected !== null
    ) {
      const receivedKeys = Object.keys(received);
      const expectedKeys = Object.keys(expected);
      if (receivedKeys.length !== expectedKeys.length) {
        return {
          pass: false,
          message: () =>
            `✗ At path '${path}': object keys length differ. Received keys: ${JSON.stringify(receivedKeys)}, expected keys: ${JSON.stringify(expectedKeys)}`,
        };
      }
      for (const key of receivedKeys) {
        if (!expected.hasOwnProperty(key)) {
          return {
            pass: false,
            message: () =>
              `✗ At path '${path}': key '${key}' found in received but not in expected`,
          };
        }
        const result = compare(received[key], expected[key], `${path}.${key}`);
        if (!result.pass) return result;
      }
    } else {
      const pass = received === expected;
      return {
        pass,
        message: () =>
          pass
            ? `✓ At path '${path}': expected ${JSON.stringify(received)} not to equal ${JSON.stringify(expected)}`
            : `✗ At path '${path}': expected ${JSON.stringify(received)} to equal ${JSON.stringify(expected)}`,
      };
    }
    return { pass: true };
  };

  return compare(received, expected);
};

/**
 * Creates an object composed of the picked object properties.
 *
 * @param {Object} o The source object.
 * @param {string[]} props The property names to pick.
 * @returns {Object} The new object with the picked properties.
 */
export function pick(o, props) {
  return Object.assign(
    {},
    ...props.map((prop) => {
      if (o[prop] !== undefined) {
        return { [prop]: o[prop] };
      }
    }),
  );
}

/**
 * Measures the execution time of an asynchronous function.
 *
 * @template T
 * @param {() => Promise<T>} fn The asynchronous function to measure.
 * @returns {Promise<{result: T, time: number}>} A promise that resolves to an object containing the result of the function and the time taken in milliseconds.
 */
export async function time(fn) {
  const start = performance.now();
  const result = await fn();
  const end = performance.now();
  return {
    result,
    time: end - start,
  };
}
