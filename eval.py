import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from exps.base_exp import BaseConfig, BaseExp, dataclass, cached_property
from exp_engine.accelerators.ddp_accelerator import DDPAccelerator
from torch.utils.data import Dataset
import math
from util_models.metrics import get_batch_metrices_new_more
from diffusers import logging
logging.set_verbosity_error()

def get_generator(random_seed):
    torch.manual_seed(int(random_seed))
    torch.cuda.manual_seed(int(random_seed))
    generator = torch.manual_seed(random_seed)
    return generator

@dataclass
class Cfg(BaseConfig):
    train_data_path: str = "./data/random100k"  
    val_data_path: str = "./data/DrawBench"  # "DrawBench", "PickaPic", "GenEval"
    pipeline: str = "SDXL"  # "SDXL", "DreamShaper", "DiT"

    train_lr_per_img: float = 1.5625e-06  # single image learning rate 
    train_batch_size_per_device: int = 40  
    val_batch_size_per_device: int = 4 
    train_max_epoch: int = 10
    train_warm_epoch: int = 5
    train_enable_wandb: bool = False
    patch_size: int = 32
    d_model: int = 2048
    d_ff: int = 2048 * 2
    n_heads: int = 1
    n_prediction_head: int = 0  # only for ablation study
    n_layers: int = 1
    dropout: float = 0.15

    train_nr_workers: int = 0
    val_nr_workers: int = 0
    cfg: float = 5.5
    use_ar: bool = True
    use_nll: bool = True

    suffix: str = "" 
    pretrained_path: str = ""


class SFTDataset(Dataset):
    def __init__(self, data_path, is_train=False, pipeline='SDXL'):

        self.latent = torch.load(data_path + "/process_latent.pt", weights_only=True)

        if pipeline == "DiT":
            self.prompts_embeds = torch.load(data_path + "/prompt_embeds_DiT.pt", map_location='cpu', weights_only=True)
        else:
            self.prompts_embeds = torch.load(data_path + "/prompt_embeds.pt", map_location='cpu', weights_only=True)

        prompts_list = []
        with open(data_path + "/prompts.txt", "r") as file:
            for line in file:
                prompts_list.append(line.strip())
        self.prompts_list = prompts_list

        print("Loading data done.")

    def __len__(self):
        return len(self.latent)  

    def __getitem__(self, idx):

        latent = self.latent[idx]

        noise_num_per_prompt = int(len(self.latent) / len(self.prompts_list))   # 40
        prompt_idx = int(idx // noise_num_per_prompt)
        noise_idx = int(idx % noise_num_per_prompt)

        prompt = self.prompts_list[prompt_idx]
        prompt_embed = self.prompts_embeds[prompt_idx]

        latent = latent.cpu()
        prompt_embed = prompt_embed.cpu()

        return latent, prompt, prompt_embed, prompt_idx, noise_idx


class NoiseARNet(torch.nn.Module):
    def __init__(
        self,
        patch_size: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        pretrained_path: str = "",
        pipeline: str = "SDXL",
        n_prediction_head: int = 0,
    ):
        super(NoiseARNet, self).__init__()
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.pipeline = pipeline

        if self.pipeline == "DiT":
            self.dim_adapter = nn.Linear(1024, 2048)

        self.latent_patch_encoder = nn.Sequential(
            nn.Linear(4 * patch_size * patch_size, 2 * patch_size * patch_size),
            nn.SiLU(),
            nn.Linear(2 * patch_size * patch_size, d_model)
        )
        self.register_buffer("learnable_start_token_emb", torch.randn(d_model) * 0.1)
        self.self_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
                for _ in range(n_layers)
            ]
        )
        self.self_attn_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.cross_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
                for _ in range(n_layers)
            ]
        )
        self.cross_attn_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    # nn.GELU(),    
                    nn.Dropout(dropout),
                )
                for _ in range(n_layers)
            ]
        )
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

 
        self.prediction_head_mu = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, 4 * patch_size * patch_size)
        )
        self.prediction_head_logvar = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, 4 * patch_size * patch_size)
        )

        # only for abliation study
        if n_prediction_head != 0:
            ffn_layers_mu = []
            for _ in range(n_prediction_head):
                ffn_layers_mu.append(nn.Linear(d_model, d_model * 2))
                ffn_layers_mu.append(nn.GELU())
                ffn_layers_mu.append(nn.Linear(d_model * 2, d_model))

            self.prediction_head_mu = nn.Sequential(
                *ffn_layers_mu, 
                nn.Linear(d_model, 4 * patch_size * patch_size) 
            )

            ffn_layers_logvar = []
            for _ in range(n_prediction_head):
                ffn_layers_logvar.append(nn.Linear(d_model, d_model * 2))
                ffn_layers_logvar.append(nn.GELU())
                ffn_layers_logvar.append(nn.Linear(d_model * 2, d_model))
                
            self.prediction_head_logvar = nn.Sequential(
                *ffn_layers_logvar,
                nn.Linear(d_model, 4 * patch_size * patch_size)
            )

        nr_patches = 128 * 128 // self.patch_size // self.patch_size
        self.seq_len = nr_patches + 1
        position = torch.arange(self.seq_len, dtype=torch.float).unsqueeze(1)
        pe = torch.zeros(1, self.seq_len, d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_encoding", pe)
        self._init_weights()

        if pretrained_path and ".pth" in pretrained_path:
            state = torch.load(pretrained_path) # , map_location=self.device)
            missing_keys, unexpected_keys = self.load_state_dict(state, strict=True)
            print("Pretrained model loaded successfully!")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                nn.init.zeros_(m.out_proj.bias)
        with torch.no_grad():
            if hasattr(self, 'prediction_head_mu'):
                final_layer = self.prediction_head_mu[-1]
                nn.init.uniform_(final_layer.weight, -0.001, 0.001)
                nn.init.zeros_(final_layer.bias)
            if hasattr(self, 'prediction_head_logvar'):
                final_layer = self.prediction_head_logvar[-1]
                nn.init.uniform_(final_layer.weight, -0.001, 0.001)
                nn.init.zeros_(final_layer.bias)

    @staticmethod
    def patchfy_to_origin(x, patch_size):
        """
        x: [b, hw/path/path, c*h*w] -> [b, c, h, w]
        """
        shp = x.shape
        x = x.reshape([shp[0], 128 // patch_size, 128 // patch_size, -1, patch_size, patch_size])
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(shp[0], -1, 128, 128)
        return x

    @staticmethod
    def origin_to_patchfy(x, patch_size):
        """
        x: [b, c, h, w] -> [b, hw/path/path, c*h*w]
        """
        b,c,h,w = x.shape
        # [64, 4, 128, 128] -> [64, 4, 16, 8, 16, 8]
        x = x.reshape([b, c, h // patch_size, patch_size, w // patch_size, patch_size])
        # [64, 4, 16, 8, 16, 8] -> [64, 16, 16, 4, 8, 8] -> [64, 256, 256]
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, -1, c * patch_size * patch_size)
        return x
    
    @staticmethod
    def sample_noise(mu, std):
        eps = torch.randn_like(std)  # [64, 4, 128, 128]
        sample = mu + eps * std
        return sample

    def forward_network(self, x, causal_mask, y):
        for layer_id in range(self.n_layers):
            x = self.self_attn_norm[layer_id](x)
            x = self.self_attn_layers[layer_id](x, x, x, attn_mask=causal_mask)[0] + x
            x = self.cross_attn_norm[layer_id](x)
            x = self.cross_attn_layers[layer_id](x, y, y)[0] + x
            x = self.ffn_norm[layer_id](x)
            x = self.ffn_layers[layer_id](x) + x
        mu = self.prediction_head_mu(x)
        logvar = self.prediction_head_logvar(x)
        return mu, logvar

    def forward(self, text_emb, x=None):

        if self.pipeline == "DiT":
            text_emb = self.dim_adapter(text_emb)

        if x is None:
            return self.inference(text_emb)

        x = self.origin_to_patchfy(x, self.patch_size)
        x = self.latent_patch_encoder(x)
        start_token_emb = self.learnable_start_token_emb[None, None, :].expand(x.shape[0], -1, -1)
        patch_sequence = torch.cat([start_token_emb, x], dim=1) + self.pos_encoding  

        causal_mask = torch.triu(torch.ones(self.seq_len, self.seq_len, device=x.device) * float("-inf"), diagonal=1)
        mu, logvar = self.forward_network(patch_sequence, causal_mask, text_emb)
        mu = self.patchfy_to_origin(mu[:, :-1, ...], self.patch_size)
        logvar = self.patchfy_to_origin(logvar[:, :-1, ...], self.patch_size)
        return mu, logvar


    def inference(self, text_emb=None):
        bs = text_emb.shape[0]
        start_token_emb = self.learnable_start_token_emb[None, None, :].expand(bs, -1, -1)
        sequence = start_token_emb

        sampled_noise_list = []
        for patch_id in range(128**2 // self.patch_size // self.patch_size):
            pos_enc = self.pos_encoding[:, : sequence.size(1), :]
            sequence_with_pos = sequence + pos_enc
            causal_mask = torch.triu(
                torch.ones(sequence.size(1), sequence.size(1), device=sequence.device) * float("-inf"), diagonal=1
            )
            mu, logvar = self.forward_network(sequence_with_pos, causal_mask, text_emb)
            std = torch.exp(0.5 * torch.clamp(logvar[:, -1, ...], -20, 10))
            sampled_noise = self.sample_noise(mu[:, -1, ...], std * 0.5)
            sampled_noise_list.append(sampled_noise)

            patch_emb = self.latent_patch_encoder(sampled_noise.reshape(bs, 1, -1))
            sequence = torch.cat([sequence, patch_emb], dim=1)
        
        return self.patchfy_to_origin(torch.stack(sampled_noise_list, dim=1), self.patch_size)


class Exp(BaseExp):
    @cached_property
    def tb_writer(self):
        tb_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tb_logs"))
        return tb_writer

    @cached_property
    def val_dataloader(self):
        ds_val = SFTDataset(data_path=self.cfg.val_data_path, is_train=False, pipeline=self.cfg.pipeline)
        sampler = torch.utils.data.distributed.DistributedSampler(
            ds_val, num_replicas=self.accelerator.num_processes, rank=self.accelerator.process_index
        )
        dl_val = torch.utils.data.DataLoader(
            ds_val,
            batch_size=self.cfg.val_batch_size_per_device,
            shuffle=False,
            num_workers=self.cfg.val_nr_workers,
            drop_last=True,
            sampler=sampler,
        )
        return dl_val

    @cached_property
    def accelerator(self):
        accelerator = DDPAccelerator()
        return accelerator

    @cached_property
    def model(self):
        cfg = self.cfg
        if cfg.use_ar:
            model = NoiseARNet(
                patch_size=cfg.patch_size,
                d_model=cfg.d_model,
                d_ff=cfg.d_ff,
                n_heads=cfg.n_heads,
                n_layers=cfg.n_layers,
                dropout=cfg.dropout,
                pretrained_path=cfg.pretrained_path,
                pipeline=cfg.pipeline,
                n_prediction_head=cfg.n_prediction_head,
            )
        else:
            assert False, "Need to set use_ar to True."
        return model

    def run(self):
        from diffusers import (
            DPMSolverMultistepScheduler,
            HunyuanDiTPipeline,
            StableDiffusionXLPipeline,
        )
        if self.cfg.pipeline == "SDXL":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                variant="fp16",
                use_safetensors=True,
                torch_dtype=torch.float16,
            ).to(self.accelerator.device)
        elif self.cfg.pipeline == "DreamShaper":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "lykon/dreamshaper-xl-v2-turbo", torch_dtype=torch.float16, variant="fp16"
            ).to(self.accelerator.device)
        else:
            pipe = HunyuanDiTPipeline.from_pretrained(
                "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", torch_dtype=torch.float16
            ).to(self.accelerator.device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe.to(self.accelerator.device)

        model = self.accelerator.prepare(self.model)
        model.to(self.accelerator.device)

        self.eval(epoch_id=0)
        
    def eval(self, epoch_id = -1):
        accelerator = self.accelerator
        model = accelerator.model
        model.eval()

        with torch.no_grad():
            torch.manual_seed(0)  

            val_imgs_path = os.path.join(self.output_dir, "inference_imgs_epoch%d" % epoch_id)
            os.makedirs(val_imgs_path, exist_ok=True)
            print("==> val image path: ", val_imgs_path)

            origin_imgs_path = os.path.join(self.cfg.val_data_path, "origin_imgs_%s"%self.cfg.pipeline)
            os.makedirs(origin_imgs_path, exist_ok=True)

            print('Start inference imgs...')
            for _, batch in enumerate(self.val_dataloader):
                _, prompts, prompt_embeds, prompt_idx, noise_idx = (_ for _ in batch)
                prompt_embeds = prompt_embeds.to(self.accelerator.device).to(torch.float32)
                pred_noise = model(prompt_embeds)

                pipe = self.pipe.to(torch.float16)
                generator = get_generator(5555)  
                golden_imgs = pipe(
                    prompt=list(prompts),
                    height=1024,
                    width=1024,
                    num_inference_steps=50,
                    guidance_scale=self.cfg.cfg,
                    latents=pred_noise.half(),
                    generator=generator,
                ).images

                for idx, item in enumerate(golden_imgs):
                    item.save(
                        f"{val_imgs_path}/{prompt_idx[idx]:08d}_{noise_idx[idx]:04d}_{prompts[idx][:40]}.jpg"
                    )

                del golden_imgs
                torch.cuda.empty_cache()
                    
        print("==> eval metrics begin")
        eval_prompts_path = os.path.join(self.cfg.val_data_path, "prompts.txt")
        (
            eval_clipscore,
            eval_hpsv2,
            eval_pickscore,
            eval_aes,
            eval_imagereward,
            eval_mps,
        ) = get_batch_metrices_new_more(
            eval_prompts_path=eval_prompts_path,
            eval_images_path=val_imgs_path,
            base_images_path=origin_imgs_path,
            device=self.accelerator.device,
        )

        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(
            f"epoch【{epoch_id}】@{nowtime} @{self.cfg.pipeline} --> clipscore= {eval_clipscore:.8f}, hpsv2= {eval_hpsv2:.8f}, pickscore= {eval_pickscore:.8f}, aes= {eval_aes:.8f}, imagereward= {eval_imagereward:.8f}, mps= {eval_mps:.8f}"
        )


if __name__ == "__main__":
    exp = Exp().set_cfg(Cfg())
    exp.accelerator.print(exp.get_cfg_as_str(exp.cfg))
    exp.run()
