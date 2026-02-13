# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
import pytest
from pathlib import Path
from PIL import Image
import tempfile

import openvino_genai

# Add paths for utilities
# TESTS_ROOT = Path(__file__).parent
# sys.path.insert(0, str(TESTS_ROOT.parent / "tools" / "who_what_benchmark"))
# sys.path.insert(0, str(TESTS_ROOT))

import whowhatbench
from utils.hugging_face import sanitize_model_id
from utils.constants import get_ov_cache_converted_models_dir, get_ov_cache_dir
from utils.atomic_download import AtomicDownloadManager
from utils.network import retry_request

try:
    from optimum.intel import OVFluxPipeline, OVStableDiffusion3Pipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

@pytest.fixture
def model_id(request):
    """Get model ID from CLI or use default"""
    return request.config.getoption("--model_id", default=None)


# def pytest_addoption(parser):
#     parser.addoption(
#         "--model_id",
#         action="store",
#         default=None,
#         help="HuggingFace model ID (e.g., stabilityai/stable-diffusion-3.5-medium)"
#     )
#     parser.addoption(
#         "--model_path",
#         action="store",
#         default=None,
#         help="Path to already converted model directory (skips download)"
#     )


def generate_image_genai(model, prompt, num_inference_steps, generator=None, empty_adapters=False):
    """Generation function for whowhatbench evaluator"""
    image_tensor = model.generate(
        prompt,
        width=512,
        height=512,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
    )

    image = Image.fromarray(image_tensor.data[0])
    return image


@pytest.mark.parametrize("model_id", ["stabilityai/stable-diffusion-3.5-medium", "black-forest-labs/FLUX.1-dev"])
def test_taylorseer_similarity_and_performance(model_id):
    device = "CPU"
    num_inference_steps = 25
    # model_path = download_and_convert_diffusion_model(model_id)

    pipe = openvino_genai.Text2ImagePipeline("/home/ltalamanova/openvino.genai/models/sd3.5-medium", device)
    # pipe = openvino_genai.Text2ImagePipeline(str(model_path), device)

    # start_time = time.time()
    # evaluator = whowhatbench.Text2ImageEvaluator(
    #     base_model=pipe,
    #     gt_data=get_ov_cache_dir() / "gt_data.cvs",
    #     # test_data=None,
    #     num_inference_steps=num_inference_steps,
    #     resolution=(512, 512),
    #     gen_image_fn=generate_image_genai,
    #     # seed=42,
    #     is_genai=True,
    # )
    # baseline_time = time.time() - start_time

    # print(f"Baseline generation completed in {baseline_time:.2f}s")
    # num_test_images = len(evaluator.gt_data)
    # print(f"  Average time per image: {baseline_time / num_test_images:.2f}s")

    print("Generating images with TaylorSeer caching...")
    ts_config = openvino_genai.TaylorSeerCacheConfig(
        3,   # cache_interval
        6,   # disable_cache_before_step (warmup)
        -2   # disable_cache_after_step
    )
    pipe.enable_feature_caching(ts_config)

    start_time = time.time()
    all_metrics_per_prompt, all_metrics = evaluator.score(
        pipe,
        gen_image_fn=generate_image_genai,
        # output_dir=str(tmp_path)
    )
    optimized_time = time.time() - start_time

    del pipe

    print(f"TaylorSeer generation completed in {optimized_time:.2f}s")
    print(f"  Average time per image: {optimized_time / num_test_images:.2f}s")

    speedup = baseline_time / optimized_time
    print(f"Speedup: {speedup:.2f}x")
    print(f"Time reduction: {((baseline_time - optimized_time) / baseline_time * 100):.1f}%")

    # Display overall metrics
    for metric_name, metric_value in all_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Display per-prompt metrics
    print("\nPer-prompt similarity scores:")
    prompts_list = evaluator.gt_data["prompts"].tolist()
    for idx, row in all_metrics_per_prompt.iterrows():
        prompt = prompts_list[idx]
        similarity = row.get('similarity', row.get('clip_score', 0))
        print(f"  Prompt {idx + 1}: {similarity:.4f}")
        print(f"    '{prompt[:60]}...'")

    # Get average similarity
    avg_similarity = all_metrics[0].get('similarity', all_metrics[0].get('clip_score', 0))

    print(f"Model: {model_id}")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Average Similarity: {avg_similarity:.4f}")

    assert speedup >= 1.2, f"Expected speedup >= 1.2x, got {speedup:.2f}x"
    assert avg_similarity >= 0.85, f"Expected similarity >= 0.85, got {avg_similarity:.4f}"


def download_and_convert_diffusion_model(model_id: str) -> Path:
    if not DIFFUSERS_AVAILABLE:
        raise ImportError("Diffusers support not available. Install with: pip install optimum[openvino,diffusers]")

    dir_name = sanitize_model_id(model_id)
    ov_cache_converted_dir = get_ov_cache_converted_models_dir()
    models_path = ov_cache_converted_dir / dir_name

    manager = AtomicDownloadManager(models_path)

    # Check if already converted
    if manager.is_complete() or (models_path / "model_index.json").exists():
        return models_path

    if "flux" in model_id.lower():
        pipeline_class = OVFluxPipeline
    elif "stable-diffusion-3" in model_id.lower():
        pipeline_class = OVStableDiffusion3Pipeline
    else:
        raise ValueError(f"Unsupported diffusion model: {model_id}. Supported: Flux and Stable Diffusion 3")

    def convert_to_temp(temp_path: Path) -> None:
        # Download and convert using optimum-intel
        pipe = retry_request(
            lambda: pipeline_class.from_pretrained(
                model_id,
                export=True,
                compile=False,
                local_files_only=False,
                load_in_8bit=False,
            )
        )
        # Save to temp directory
        pipe.save_pretrained(str(temp_path))

    manager.execute(convert_to_temp)
    return models_path


# if __name__ == "__main__":
#     # Allow running directly for quick testing
#     import argparse
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_id", help="HuggingFace model ID")
#     parser.add_argument("--model_name", default="sd3.5", choices=["sd3.5", "flux"])
#     args = parser.parse_args()
    
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         tmp_path = Path(tmp_dir)
        
#         # Mock the model_id fixture
#         class MockRequest:
#             class Config:
#                 @staticmethod
#                 def getoption(name, default=None):
#                     if name == "--model_id":
#                         return args.model_id
#                     return default
#             config = Config()
        
#         test_taylorseer_similarity_and_performance(
#             args.model_name, 
#             args.model_id,
#             tmp_path
#         )
