# 🤗 Transformers.js Benchmarking

A versatile toolkit designed to measure and compare the performance of Transformers.js/ONNX models across various environments, including: in-browser (WASM, WebGPU, WebNN), Node.js, Bun, or Deno.

## Key Features

- **Multi-Platform Support:** Benchmark models directly in-browser (WASM, WebGPU, WebNN) as well as in server-side environments like Node.js, Bun, and Deno.
- **Efficient Model Ops Collection:** Streaming-based collection functionality to efficiently gather model operations without needing to download entire models.

## Getting Started

1. **Install core dependencies** first

   ```sh
   cd packages/core/
   npm i
   ```

   Then run with one of the following:

2. **Run Benchmarks**

   - web ([online demo](https://huggingface.co/spaces/onnx-internal-testing/transformers.js-benchmarking)):

     ```sh
     cd packages/web/
     npm i
     npm run dev
     ```

   - Node.js

     ```sh
     cd packages/node/
     npm i
     node index.js
     ```

   - Bun
     ```sh
     cd packages/bun/
     bun install
     bun run index.ts
     ```

3. **(Optional) Prepare model operations**

   - Install Python dependencies:

     ```sh
     python3 -m venv .venv
     source .venv/bin/activate
     python3 -m pip install -r scripts/requirements.txt
     ```

   - Clone model-explorer dataset

     ```sh
     npm run pull:model-explorer
     ```

   - Build model operations with:

     ```sh
     npm run build:architectures
     ```

## Repository Structure

```
├── packages
│   ├── core         # Core engine powering the benchmarking suite.
│   ├── web          # User-friendly web interface for running benchmarks.
│   ├── node         # CLI tailored for Node.js environments.
│   └── bun          # CLI support specifically built for Bun.
├── scripts          # Utility scripts for tasks like collecting model operations.
└── data             # Repository and model operation files (e.g., model_ops.csv).
```

## Additional Resources

For more details on available models and further information, check out the models on [Hugging Face](https://huggingface.co/models?library=transformers.js).
