import gradio as gr
import torch
import random
import math
import os
import numpy as np
import torchvision.transforms as T

from omegaconf import OmegaConf
from safetensors import safe_open
from peft import LoraConfig
import peft

from omnitry.models.transformer_flux import FluxTransformer2DModel
from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline
from huggingface_hub import snapshot_download


# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16           # 🔥 T4 friendly
MAX_AREA = 768 * 768            # 🔥 VRAM safe
DEFAULT_STEPS = 15
DEFAULT_GUIDANCE = 20


# =========================
# DOWNLOAD WEIGHTS
# =========================
snapshot_download(
    repo_id="Kunbyte/OmniTry",
    local_dir="./OmniTry",
    local_dir_use_symlinks=False
)

args = OmegaConf.load("configs/omnitry_v1_unified.yaml")


# =========================
# LOAD MODEL
# =========================
transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev",
    subfolder="transformer",
    torch_dtype=DTYPE
).to(DEVICE).requires_grad_(False)

pipeline = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev",
    transformer=transformer,
    torch_dtype=DTYPE
).to(DEVICE)

# 🔥 MEMORY OPTIMIZATION
pipeline.enable_attention_slicing("max")
pipeline.enable_vae_slicing()
pipeline.enable_vae_tiling()
torch.backends.cuda.matmul.allow_tf32 = True


# =========================
# LORA SETUP
# =========================
lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    init_lora_weights="gaussian",
    target_modules=[
        'x_embedder',
        'attn.to_k', 'attn.to_q', 'attn.to_v', 'attn.to_out.0',
        'attn.add_k_proj', 'attn.add_q_proj', 'attn.add_v_proj',
        'attn.to_add_out',
        'ff.net.0.proj', 'ff.net.2',
        'ff_context.net.0.proj', 'ff_context.net.2',
        'norm1_context.linear', 'norm1.linear',
        'norm.linear', 'proj_mlp', 'proj_out'
    ]
)

transformer.add_adapter(lora_config, adapter_name="vtryon_lora")
transformer.add_adapter(lora_config, adapter_name="garment_lora")

with safe_open("OmniTry/omnitry_v1_unified.safetensors", framework="pt") as f:
    lora_weights = {k: f.get_tensor(k) for k in f.keys()}
    transformer.load_state_dict(lora_weights, strict=False)


# =========================
# LORA HACK (UNCHANGED LOGIC)
# =========================
def create_hacked_forward(module):

    def lora_forward(self, active_adapter, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        if active_adapter is not None:
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)
            result = result + lora_B(lora_A(x)) * scaling
        return result

    def hacked_forward(self, x, *args, **kwargs):
        return torch.cat((
            lora_forward(self, "vtryon_lora", x[:1], *args, **kwargs),
            lora_forward(self, "garment_lora", x[1:], *args, **kwargs),
        ), dim=0)

    return hacked_forward.__get__(module, type(module))


for _, m in transformer.named_modules():
    if isinstance(m, peft.tuners.lora.layer.Linear):
        m.forward = create_hacked_forward(m)


# =========================
# UTILS
# =========================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# INFERENCE
# =========================
def generate(person_image, object_image, object_class, steps, guidance, seed):

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    seed_everything(seed)

    oW, oH = person_image.width, person_image.height
    ratio = min(1.0, math.sqrt(MAX_AREA / (oW * oH)))
    tW = int(oW * ratio) // 16 * 16
    tH = int(oH * ratio) // 16 * 16

    transform = T.Compose([
        T.Resize((tH, tW)),
        T.ToTensor()
    ])

    person = transform(person_image)

    ratio = min(tW / object_image.width, tH / object_image.height)
    obj = T.Resize((
        int(object_image.height * ratio),
        int(object_image.width * ratio)
    ))(object_image)
    obj = T.ToTensor()(obj)

    padded = torch.ones_like(person)
    y = (tH - obj.shape[1]) // 2
    x = (tW - obj.shape[2]) // 2
    padded[:, y:y+obj.shape[1], x:x+obj.shape[2]] = obj

    img_cond = torch.stack([person, padded]).to(
        device=DEVICE,
        dtype=DTYPE
    )

    mask = torch.zeros_like(img_cond)

    with torch.no_grad():
        out = pipeline(
            prompt=[args.object_map[object_class]] * 2,
            height=tH,
            width=tW,
            img_cond=img_cond,
            mask=mask,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator(DEVICE).manual_seed(seed)
        ).images[0]

    torch.cuda.empty_cache()
    return out


# =========================
# GRADIO UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("# 🧥 OmniTry – T4 Optimized")

    with gr.Row():
        person = gr.Image(type="pil", label="Person")
        garment = gr.Image(type="pil", label="Garment")
        output = gr.Image(type="pil", label="Result")

    cls = gr.Dropdown(
        choices=args.object_map.keys(),
        label="Object Class"
    )

    with gr.Accordion("Advanced", open=False):
        steps = gr.Slider(1, 30, value=DEFAULT_STEPS, step=1)
        guidance = gr.Slider(1, 30, value=DEFAULT_GUIDANCE, step=0.5)
        seed = gr.Number(value=-1, precision=0)

    btn = gr.Button("Try On")

    btn.click(
        generate,
        inputs=[person, garment, cls, steps, guidance, seed],
        outputs=output
    )

demo.launch(server_name="0.0.0.0", share=True)
