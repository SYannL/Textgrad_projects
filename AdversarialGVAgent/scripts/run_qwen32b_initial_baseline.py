#!/usr/bin/env python
"""Run the Qwen3-32B one-pass baseline with the shared initial prompt."""

from adversarial_gv.initial_baseline import run


if __name__ == "__main__":
    run(
        default_model="qwen32b-api",
        default_base_url="http://127.0.0.1:8000/v1",
        default_output_dir="runs/qwen32b_initial_prompt_baseline",
    )

