import torch
from diffusers import StableDiffusionPipeline


def get_safe_device():
    try:
        if torch.cuda.is_available():
            torch.cuda.current_device()
            return "cuda"
    except RuntimeError:
        pass
    return "cpu"


device = get_safe_device()
torch_dtype = torch.float16 if device == "cuda" else torch.float32
pipeline_kwargs = {"torch_dtype": torch_dtype}
if device == "cuda":
    pipeline_kwargs["variant"] = "fp16"

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    **pipeline_kwargs,
)
pipe = pipe.to(device)
prompt = "A white cat sitting on a mat"
image = pipe(prompt).images[0]
image.save("white_cat.png")
