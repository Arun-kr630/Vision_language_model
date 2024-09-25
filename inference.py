from PIL import Image
import torch
import fire

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model

def move_inputs_to_device(model_inputs, device):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs

def get_model_inputs(processor, prompt, image_file_path, device):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs

def test_inference(model: PaliGemmaForConditionalGeneration,processor: PaliGemmaProcessor,device: str,prompt: str,
    image_file_path: str,max_tokens_to_generate: int,):


    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []
    for _ in range(max_tokens_to_generate):
        outputs = model(input_ids=input_ids,pixel_values=pixel_values,attention_mask=attention_mask,kv_cache=kv_cache,)
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)
        generated_tokens.append(next_token)
        if next_token.item() == stop_token:
            break
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1)

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(prompt + decoded)

def main():
    model_path="D:\project\models--google--paligemma-3b-pt-224\snapshots\weights"
    prompt="this building is "
    image_file_path="./test_images/pic1.jpg"
    max_tokens_to_generate: int = 100,
    only_cpu: bool = False,

    device = "cpu"

    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in use: ", device)

    print(f"Loading model")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference")
    with torch.no_grad():
        test_inference(model,processor,device,prompt,image_file_path,max_tokens_to_generate)


if __name__ == "__main__":
    fire.Fire(main)