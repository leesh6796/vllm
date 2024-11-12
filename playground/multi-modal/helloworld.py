import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = "/mnt/shared/models/llama/Llama3.2-11B-Vision-Instruct"

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

model = MllamaForConditionalGeneration.from_pretrained(
    model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)
print(processor)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "If I had to write a haiku for this one, it would be: ",
            },
        ],
    }
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(
    model.device
)

output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))