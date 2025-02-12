import os
import argparse
import re
import time
import json
import logging
from pathlib import Path
from typing import Set, Dict, Tuple, List, Any, Callable

from huggingface_hub import HfApi, hf_hub_url, HfFileSystem
import onnx
from tqdm import tqdm

from parser import stream_parse_model_header

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ALLOWED_AUTHORS = [
    "hf-internal-testing",
    "onnx-internal-testing",
    "Xenova",
    "onnx-community",
    "distil-whisper",
    "HuggingFaceTB",
    "microsoft",
    "mixedbread-ai",
    "Mozilla",
    "nomic-ai",
    "jinaai",
    "lightonai",
    "llava-hf",
    "Marqo",
    "Snowflake",
]
BANNED_REPOS = set()

CACHE_DIR = Path(__file__).parent / ".cache"
ALLOWED_QUANTIZATIONS = ["fp16", "uint8", "int8", "quantized", "q4", "q4f16", "bnb4"]
DISALLOWED_FILE_PATTERNS = [
    re.compile(r'decoder(_with_past)?_model(?!_merged)'),
]
ALLOWED_QUANTIZATION_PATTERNS = re.compile(
    r'^(.+?)(?:_(' + "|".join(ALLOWED_QUANTIZATIONS) + r'))?\.onnx$'
)
ALLOWED_REPO_PATTERN = re.compile(r"tiny-random-\w+(?:For\w+|Model)");

def get_operators(model: onnx.ModelProto) -> Set[str]:
    """
    Recursively traverses the ONNX graph and returns a set of operator names.

    Args:
        model: Loaded ONNX model.

    Returns:
        Set of operator names.
    """
    operators: Set[str] = set()

    def traverse_graph(graph: onnx.GraphProto):
        for node in graph.node:
            operators.add(node.op_type)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    traverse_graph(attr.g)

    traverse_graph(model.graph)
    return operators

def retry_operation(func: Callable, *args, max_retries: int = 5, initial_delay: int = 1, **kwargs) -> Any:
    """
    Retry a given operation with exponential backoff.

    Args:
        func: Callable function to execute.
        *args: Positional arguments for the function.
        max_retries: Maximum retry attempts.
        initial_delay: Initial delay between retries.
        **kwargs: Keyword arguments for the function.

    Returns:
        The function result or False if a specific RuntimeError occurs.

    Raises:
        Exception if max retries are exceeded or on non-retryable errors.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except RuntimeError:
            return False
        except Exception as e:
            status = getattr(e, "response", None)
            if status and (status.status_code == 429 or 500 <= status.status_code < 600):
                logging.warning(
                    f"Attempt {attempt + 1} failed with status {status.status_code}. Retrying after {delay} seconds..."
                )
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise Exception("Max retries exceeded while trying to execute operation.")

def collect_model_ops(
    model_limit: int = None,
    use_cache: bool = True,
    limit: int = 500,
    include_all_models: bool = False,
) -> None:
    """
    Collects the operators used by ONNX models from the Hugging Face Hub, downloads models as needed,
    and generates JavaScript files with the aggregated model metadata.

    Args:
        model_limit: Maximum number of models to query from the Hub.
        use_cache: Whether to use the local cache folder for saving/loading models.
        limit: Maximum number of unique models to process.
        include_all_models: If True, also includes models from transformers.js library.
    """
    api = HfApi()
    fs = HfFileSystem()

    logging.info("Collecting models from the Hugging Face Hub...")
    onnx_models = list(
        api.list_models(
            library="onnx", search="tiny-random", limit=model_limit, sort="downloads",
            direction=-1, fetch_config=True
        )
    )
    if include_all_models:
        tfjs_models = list(
            api.list_models(
                library="transformers.js", limit=model_limit, sort="downloads",
                direction=-1, fetch_config=True
            )
        )
        onnx_models.extend(tfjs_models)

    logging.info("Found %d models.", len(onnx_models))
    unique_models: Dict[str, Any] = {}
    model_types: Dict[str, str] = {}
    model_architectures: Dict[str, str] = {}

    for model in onnx_models:
        repo_id = model.modelId
        if model.private or model.gated:
            logging.info("Skipping private or gated model: %s", repo_id)
            continue
        author = repo_id.split('/')[0]
        if author not in ALLOWED_AUTHORS:
            logging.info("Skipping unauthorized author: %s", repo_id)
            continue
        if repo_id in BANNED_REPOS:
            logging.info("Skipping banned model: %s", repo_id)
            continue
        if not ALLOWED_REPO_PATTERN.search(repo_id):
            logging.info("Skipping disallowed model: %s", repo_id)
            continue

        unique_models[repo_id] = model
        cfg = getattr(model, "config", {}) or {}
        model_types[repo_id] = cfg.get("model_type", "unknown")
        model_architectures[repo_id] = cfg.get("architectures", [])

    # Limit unique models based on downloads
    unique_models = dict(
        sorted(unique_models.items(), key=lambda x: x[1].downloads, reverse=True)[:limit]
    )
    logging.info("Processing %d unique models.", len(unique_models))

    processed_repo_list_path = CACHE_DIR / "models.txt"
    processed_repos = set()
    if use_cache and processed_repo_list_path.exists():
        with processed_repo_list_path.open("r") as f:
            processed_repos = {line.strip() for line in f}

    model_type_ops: Dict[Tuple[str, str, str], Set[str]] = {}
    for repo_id, model in tqdm(unique_models.items(), desc="Processing Models"):
        processed = repo_id in processed_repos
        if processed:
            model_cache_folder = CACHE_DIR / repo_id
            files = []
            if model_cache_folder.exists():
                files = [f"{repo_id}/onnx/{f}" for f in os.listdir(model_cache_folder)]
        else:
            pattern = f"{repo_id}/**/*.onnx"
            result = retry_operation(fs.glob, pattern, detail=True)
            files = list(result.keys()) if result else []

        for file_path in files:
            relative_path = os.path.relpath(file_path, repo_id)
            if not relative_path.startswith("onnx/"):
                continue
            subfolder, file_name = os.path.split(relative_path)
            match = ALLOWED_QUANTIZATION_PATTERNS.match(file_name)
            if not match:
                continue
            if any(p.search(file_name) for p in DISALLOWED_FILE_PATTERNS):
                logging.info("Skipping disallowed file: %s/%s/%s", repo_id, subfolder, file_name)
                continue

            quantization = match.group(2) or "fp32"
            model_proto = None
            cache_path = None
            if use_cache:
                cache_folder = CACHE_DIR / repo_id
                cache_folder.mkdir(exist_ok=True)
                cache_path = cache_folder / file_name
                if cache_path.exists():
                    model_proto = onnx.load(str(cache_path), load_external_data=False)

            if model_proto is None:
                logging.info('Downloading model "%s/%s/%s"', repo_id, subfolder, file_name)
                url = hf_hub_url(repo_id=repo_id, subfolder=subfolder, filename=file_name)
                model_proto = retry_operation(stream_parse_model_header, url)
                if use_cache and cache_path and model_proto:
                    onnx.save(model_proto, str(cache_path))

            if model_proto:
                ops_set = get_operators(model_proto)
                m_type = model_types.get(repo_id, "unknown")
                key = (m_type, repo_id, quantization)
                if key not in model_type_ops:
                    model_type_ops[key] = set()
                model_type_ops[key].update(ops_set)

        if use_cache and not processed:
            with processed_repo_list_path.open("a") as f:
                f.write(repo_id + "\n")
            processed_repos.add(repo_id)

    architecture_ops: Dict[str, List[Tuple[str, str, Set[str]]]] = {}
    for (m_type, model_id, q), ops_set in model_type_ops.items():
        if m_type not in architecture_ops:
            architecture_ops[m_type] = []
        else:
            # Avoid duplicating operations already covered
            if any(existing_ops == ops_set for _, _, existing_ops in architecture_ops[m_type]):
                continue
        architecture_ops[m_type].append((model_id, q, ops_set))

    # Generate JS files
    core_dir = Path(__file__).parent.parent / "packages/core/src"
    arch_dir = core_dir / "architectures"
    arch_dir.mkdir(parents=True, exist_ok=True)

    for m_type, model_list in architecture_ops.items():
        js_path = arch_dir / f"{m_type}.js"
        models_data = [
            {"model_id": model_id, "dtype": quantization, "architectures": model_architectures[model_id], "ops": sorted(list(ops))}
            for model_id, quantization, ops in model_list
        ]
        template = (
            f"// NOTE: This file has been auto-generated. Do not edit directly.\n\n"
            f"export default {{ model_type: '{m_type}', models: {json.dumps(models_data)} }}\n"
        )
        with js_path.open("w") as f:
            f.write(template)

    arch_index_path = core_dir / "architectures.js"
    with arch_index_path.open("w") as fp:
        fp.write("// NOTE: This file has been auto-generated. Do not edit directly.\n")
        for m_type in architecture_ops.keys():
            safe_m_type = m_type.replace('-', '_')
            fp.write(f"export {{ default as {safe_m_type} }} from './architectures/{m_type}.js';\n")

def main() -> None:
    """
    Parses command line arguments and initiates the collection of model operators.
    """
    parser = argparse.ArgumentParser(
        description="Collect operators used in ONNX models from the Hugging Face Hub."
    )
    parser.add_argument("--model_limit", type=int, default=None, help="Maximum number of models to query from the Hub.")
    parser.add_argument("--limit", type=int, default=500, help="Maximum number of unique models to process.")
    parser.add_argument(
        "--disable_cache", action="store_true",
        help="Disable using local cache for loading/saving models."
    )
    parser.add_argument(
        "--all_models", action="store_true",
        help="Include models from the transformers.js library in addition to tiny-random models."
    )
    args = parser.parse_args()

    collect_model_ops(
        model_limit=args.model_limit,
        use_cache=not args.disable_cache,
        limit=args.limit,
        include_all_models=args.all_models,
    )

if __name__ == "__main__":
    main()
