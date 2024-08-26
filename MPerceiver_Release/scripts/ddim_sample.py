"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
# from ldm.models.diffusion.ddim import DDIMSampler
# from ldm.models.diffusion.plms import PLMSSampler
import math
import copy
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization
from ldm.models.diffusion.ddim_ir import DDIMSampler

def chunk(it, size):
	it = iter(it)
	return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
	print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu")
	if "global_step" in pl_sd:
		print(f"Global Step: {pl_sd['global_step']}")
	sd = pl_sd["state_dict"]
	model = instantiate_from_config(config.model)
	m, u = model.load_state_dict(sd, strict=False)
	if len(m) > 0 and verbose:
		print("missing keys:")
		print(m)
	if len(u) > 0 and verbose:
		print("unexpected keys:")
		print(u)

	model.cuda()
	model.eval()
	return model

def load_img(path):
	image = Image.open(path).convert("RGB")
	w, h = image.size
	print(f"loaded input image of size ({w}, {h}) from {path}")
	w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
	image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = np.array(image).astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return 2.*image - 1.


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--indir",
		type=str,
		nargs="?",
		help="path to the input image",
		default="input/",
	)
	parser.add_argument(
		"--outdir",
		type=str,
		nargs="?",
		help="dir to write results to",
		default="output/",
	)
	parser.add_argument(
		"--ddim_steps",
		type=int,
		default=200,
		help="number of ddim sampling steps",
	)
	parser.add_argument(
		"--C",
		type=int,
		default=4,
		help="latent channels",
	)
	parser.add_argument(
		"--f",
		type=int,
		default=8,
		help="downsampling factor, most often 8 or 16",
	)
	parser.add_argument(
		"--n_samples",
		type=int,
		default=1,
		help="how many samples to produce for each given prompt. A.k.a batch size",
	)
	parser.add_argument(
		"--config",
		type=str,
		default="",
		help="path to config which constructs model",
	)
	parser.add_argument(
		"--ckpt",
		type=str,
		default="",
		help="path to checkpoint of model",
	)
	parser.add_argument(
		"--vae_ckpt",
		type=str,
		default="ckpt/vae_21.ckpt",
		help="path to checkpoint of vae model",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="the seed (for reproducible sampling)",
	)
	parser.add_argument(
		"--precision",
		type=str,
		help="evaluate at this precision",
		choices=["full", "autocast"],
		default="autocast"
	)
	parser.add_argument(
		"--input_size",
		type=int,
		default=512,
		help="input size",
	)
	parser.add_argument(
		"--dec_w",
		type=float,
		default=0.5,
		help="weight for combining VQGAN and Diffusion",
	)
	parser.add_argument(
		"--colorfix_type",
		type=str,
		default="none",
		help="Color fix type to adjust the color of HR result according to LR input: adain (used in paper); wavelet; nofix",
	)
	parser.add_argument(
		"--ddim_eta",
		type=float,
		default=0.0,
	)
	parser.add_argument(
		"--strength",
		type=float,
		default=1.0,
	)
    
	opt = parser.parse_args()
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	print('>>>>>>>>>>color correction>>>>>>>>>>>')
	if opt.colorfix_type == 'adain':
		print('Use adain color correction')
	elif opt.colorfix_type == 'wavelet':
		print('Use wavelet color correction')
	else:
		print('No color correction')
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

	# vqgan_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
	# vq_model = load_model_from_config(vqgan_config, opt.vae_ckpt)
	# vq_model = vq_model.to(device)
	# vq_model.decoder.fusion_w = opt.dec_w

	seed_everything(opt.seed)

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(opt.input_size),
		torchvision.transforms.CenterCrop(opt.input_size),
	])

	config = OmegaConf.load(f"{opt.config}")
	model = load_model_from_config(config, f"{opt.ckpt}")
	model = model.to(device)
	
	sampler = DDIMSampler(model)
	sampler.make_schedule(ddim_num_steps=opt.ddim_steps,ddim_eta=opt.ddim_eta,verbose=False)


	os.makedirs(opt.outdir, exist_ok=True)
	outpath = opt.outdir

	batch_size = opt.n_samples

	img_list_ori = os.listdir(opt.indir)
	img_list = copy.deepcopy(img_list_ori)
	init_image_list = []
	for item in img_list_ori:
		if os.path.exists(os.path.join(outpath, item)):
			img_list.remove(item)
			continue
		cur_image = load_img(os.path.join(opt.indir, item)).to(device)
		# cur_image = transform(cur_image)
		cur_image = cur_image.clamp(-1, 1)
		init_image_list.append(cur_image)
	init_image_list = torch.cat(init_image_list, dim=0)
	niters = math.ceil(init_image_list.size(0) / batch_size)
	init_image_list = init_image_list.chunk(niters)
	# model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
	# 					  linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
	# model.num_timesteps = 1000

	# sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
	# sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

	# use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
	# last_alpha_cumprod = 1.0
	# new_betas = []
	# timestep_map = []
	# for i, alpha_cumprod in enumerate(model.alphas_cumprod):
	# 	if i in use_timesteps:
	# 		new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
	# 		last_alpha_cumprod = alpha_cumprod
	# 		timestep_map.append(i)
	# new_betas = [beta.data.cpu().numpy() for beta in new_betas]
	# model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
	# model.num_timesteps = 1000
	# model.ori_timesteps = list(use_timesteps)
	# model.ori_timesteps.sort()
	# model = model.to(device)
    
        
	precision_scope = autocast if opt.precision == "autocast" else nullcontext
	niqe_list = []
	t_enc = int(opt.strength*opt.ddim_steps)
	with torch.no_grad():
		with precision_scope("cuda"):
			with model.ema_scope():
				tic = time.time()
				all_samples = list()
				for n in trange(niters, desc="Sampling"):
					init_image = init_image_list[n]
					# init_latent_generator, enc_fea_lq = vq_model.encode(init_image)
					# init_latent = model.get_first_stage_encoding(init_latent_generator)
					init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
					text_init = ['']*init_image.size(0)
					semantic_c = model.get_learned_conditioning(text_init)
					
					z_enc = sampler.stochastic_encode(init_latent, torch.tensor([(t_enc-1)*init_image.size(0)]).to(device))
					z_enc = torch.randn(z_enc.shape,device=z_enc.device)
                    
					# noise = torch.randn_like(init_latent)
					# If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
					# t = repeat(torch.tensor([499]), '1 -> b', b=init_image.size(0))
					# t = t.to(device).long()
					# x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
					# x_T = None
                    
					# samples, _ = model.sample(cond=semantic_c, struct_cond=init_latent, batch_size=init_image.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True)
					# x_samples = model.decode_first_stage(samples)
					x_samples = sampler.decode(z_enc,semantic_c,init_latent,t_enc,1,semantic_c)
					x_samples = model.decode_first_stage(x_samples)
					# x_samples = vq_model.decode(x_samples * 1. / model.scale_factor, enc_fea_lq)

					if opt.colorfix_type == 'adain':
						x_samples = adaptive_instance_normalization(x_samples, init_image)
					elif opt.colorfix_type == 'wavelet':
						x_samples = wavelet_reconstruction(x_samples, init_image)
					x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

					for i in range(init_image.size(0)):
						img_name = img_list.pop(0)
						basename = os.path.splitext(os.path.basename(img_name))[0]
						x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
						Image.fromarray(x_sample.astype(np.uint8)).save(
							os.path.join(outpath, basename+'.png'))

				toc = time.time()

	print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
		  f" \nEnjoy.")


if __name__ == "__main__":
	main()
	# print(space_timesteps(1000,'ddim200'))
