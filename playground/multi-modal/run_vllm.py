from transformers import AutoTokenizer
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser

from pymemorial import *


# LLama 3.2
def run_mllama(question: str, modality: str):
    assert modality == "image"

    # model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model_name = "/mnt/shared/models/llama/Llama3.2-11B-Vision-Instruct"

    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (131072) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # The configuration below has been confirmed to launch on a single L40 GPU.
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=16,
        enforce_eager=True,
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        gpu_memory_utilization=0.9,
        worker_use_ray=True,
    )

    prompt = f"<|image|><|begin_of_text|>{question}"
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# InternVL
def run_internvl(question: str, modality: str):
    assert modality == "image"

    # model_name = "OpenGVLab/InternVL2-2B"
    model_name = "/mnt/shared/models/intern/InternVL2-26B"

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        enforce_eager=True,
        pipeline_parallel_size=4,
        # tensor_parallel_size=4,
        worker_use_ray=True,
        gpu_memory_utilization=0.9,
        max_num_seqs=8,
        max_num_batched_tokens=4096,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [{"role": "user", "content": f"<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B#service
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    return llm, prompt, stop_token_ids


def get_multi_modal_input(modality: str):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if modality == "image":
        # Input image and question
        # image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
        image = Image.open("./taylor.jpg").convert("RGB")

        img_question = "What is the content of this image?"

        return {
            "data": image,
            "question": img_question,
        }

    # if args.modality == "video":
    #     # Input video and question
    #     video = VideoAsset(
    #         name="sample_demo_1.mp4", num_frames=args.num_frames
    #     ).np_ndarrays
    #     vid_question = "Why is this video funny?"

    #     return {
    #         "data": video,
    #         "question": vid_question,
    #     }

    msg = f"Modality {modality} is not supported."
    raise ValueError(msg)


def main(modality: str, num_prompts: int = 1):
    model = "internvl"
    MProfile().set_flag_profile()

    mm_input = get_multi_modal_input(modality)
    data = mm_input["data"]
    question = mm_input["question"]

    # llm, prompt, stop_token_ids = run_mllama(question, modality)
    llm, prompt, stop_token_ids = run_internvl(question, modality)

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(
        temperature=0.2, ignore_eos=True, max_tokens=1, stop_token_ids=stop_token_ids
    )

    assert num_prompts > 0
    if num_prompts == 1:
        # Single inference
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {modality: data},
        }

    else:
        # Batch inference
        inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {modality: data},
            }
            for _ in range(num_prompts)
        ]

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
