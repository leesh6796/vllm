from run_vllm import main
import os

if __name__ == "__main__":
    # cd = "cd /mnt/llm-playground/vllm/playground/multi-modal"
    # env = "/usr/bin/env /mnt/venv/conda-modal/bin/python"
    # run_command(
    #     f"{cd}; {env} /mnt/llm-playground/vllm/playground/multi-modal/run-vllm.py --modality image --num-prompts 1"
    # )

    os.chdir("/mnt/llm-playground/vllm/playground/multi-modal")
    main(modality="image", num_prompts=1)
