export const DEFAULT_NUM_WARMUP_RUNS = 5;
export const DEFAULT_NUM_RUNS = 50;

export class Test {
  constructor(config) {
    this.config = config;
  }

  async run() {
    throw new Error("Not implemented");
  }
}
