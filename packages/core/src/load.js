import * as ARCHICTURES from "./architectures.js";
import * as TRANSFORMERS_JS from "@huggingface/transformers";
import { DEFAULT_MODEL_OPTIONS } from "./defaults.js";

// console.log(ARCHICTURES);
const UNSUPPORTED_ARCHITECTURES = [];
for (const [key, value] of Object.entries(ARCHICTURES)) {
  for (const model of value.models) {
    if (model.skip) continue;

    if (model.architectures.length !== 1) {
      throw new Error(
        `Expected exactly one architecture for ${model.model_id}`,
      );
    }
    const architecture = model.architectures[0];
    const cls = TRANSFORMERS_JS[architecture];
    if (!cls) {
      console.warn(
        `Unknown architecture: ${architecture} in ${model.model_id}`,
      );
      UNSUPPORTED_ARCHITECTURES.push(architecture);
      continue;
    }
    console.log(model.model_id);
    const m = await cls
      .from_pretrained(model.model_id, {
        ...DEFAULT_MODEL_OPTIONS,
        dtype: model.dtype === "quantized" ? "q8" : model.dtype,
      })
      .catch((err) => {
        console.error(err);
        return null;
      });
    await m?.dispose();
  }
}
