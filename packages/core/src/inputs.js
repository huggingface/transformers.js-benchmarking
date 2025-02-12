/**
 * Since we want the benchmark to run in JavaScript "engine shells" (e.g., d8 for V8 or jsc for JavaScriptCore, not full browsers),
 * many browser APIs are not available, e.g., for audio processing, TextDecoder/TextEncoder, or even web workers.
 *
 * For this reason, we prepare inputs that can be passed to the models.
 */
import { RawImage } from "@huggingface/transformers";

export const DUMMY_AUDIO = new Float32Array(16000);

export const DUMMY_IMAGE = new RawImage(
  new Uint8ClampedArray(224 * 224 * 4),
  224,
  224,
  4,
);

export const TINY_DUMMY_IMAGE = new RawImage(
  new Uint8ClampedArray(64 * 32 * 3),
  64,
  32,
  3,
);
