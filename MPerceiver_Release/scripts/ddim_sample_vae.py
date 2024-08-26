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
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize,CenterCrop
import torchvision.transforms as transforms


from ldm.util import instantiate_from_config
# from ldm.models.diffusion.ddim import DDIMSampler
# from ldm.models.diffusion.plms import PLMSSampler
import math
import copy
from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization
from ldm.models.diffusion.ddim_ir import DDIMSampler
import cv2

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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

def _convert_to_rgb(image):
    return image.convert('RGB')

def m_pad(x):
    if x < 512:
        pad = 512 - x
    else:
        if x%64 == 0:
            pad = 0
        else:
            pad = 64 - x%64
    return pad

def load_img(path):
	# image = Image.open(path).convert("RGB")
	# w, h = image.size
	# print(f"loaded input image of size ({w}, {h}) from {path}")
	# w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
	# image = image.resize((w, h), resample=PIL.Image.LANCZOS)
	image = cv2.imread(path)
	h, w, c = image.shape
	pad_h = m_pad(h)
	pad_w = m_pad(w)
	image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
	#将image转为RGB格式
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image.astype(np.float32) / 255.0
	image = image[None].transpose(0, 3, 1, 2)
	image = torch.from_numpy(image)
	return (2.*image - 1.).clamp(-1, 1),h,w


def load_img_224(path):
    train_transforms = transforms.Compose([transforms.Resize((224,224),
    interpolation=InterpolationMode.BICUBIC),
	_convert_to_rgb,
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    # image = Image.open(path)
    image = cv2.imread(path)
    h, w, c = image.shape
    pad_h = m_pad(h)
    pad_w = m_pad(w)
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image.astype(np.uint8))
    return train_transforms(image).unsqueeze(0)

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--indir",
		type=str,
		nargs="?",
		help="path to the input image",
		default="",
	)
	parser.add_argument(
		"--outdir_sample",
		type=str,
		nargs="?",
		help="dir to write results to",
		default="",
	)
	parser.add_argument(
		"--ddim_steps",
		type=int,
		default=50,
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
		default="",
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

	vqgan_config = OmegaConf.load("configs/autoencoder/ae_resi8.yaml")
	vq_model = load_model_from_config(vqgan_config, opt.vae_ckpt)
	vq_model = vq_model.to(device)

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


	# os.makedirs(opt.outdir_npy, exist_ok=True)
	os.makedirs(opt.outdir_sample, exist_ok=True)
	# outpath_npy = opt.outdir_npy
	outpath_sample = opt.outdir_sample

	batch_size = opt.n_samples

	img_list_ori = os.listdir(opt.indir)
	img_list = copy.deepcopy(img_list_ori)

	precision_scope = autocast if opt.precision == "autocast" else nullcontext
	niqe_list = []
	t_enc = int(opt.strength*opt.ddim_steps)
	bz = 1
	niters = math.ceil(len(img_list_ori) / bz)
	with torch.no_grad():
		with precision_scope("cuda"):
			with model.ema_scope():
				tic = time.time()
				all_samples = list()
				for n in trange(niters, desc="Sampling"):
					if n == niters - 1:
						init_image = [load_img(os.path.join(opt.indir, item))[0].to(device) for item in img_list[n*bz:]]
						h_list = [load_img(os.path.join(opt.indir, item))[1] for item in img_list[n*bz:]]
						w_list = [load_img(os.path.join(opt.indir, item))[2] for item in img_list[n*bz:]]
						init_image_224 = [load_img_224(os.path.join(opt.indir, item)).to(device) for item in img_list[n*bz:]]
						init_image = torch.cat(init_image, dim=0)
						init_image_224 = torch.cat(init_image_224, dim=0)
					else:
						init_image = [load_img(os.path.join(opt.indir, item))[0].to(device) for item in img_list[n*bz:(n+1)*bz]]
						init_image_224 = [load_img_224(os.path.join(opt.indir, item)).to(device) for item in img_list[n*bz:(n+1)*bz]]
						h_list = [load_img(os.path.join(opt.indir, item))[1] for item in img_list[n*bz:(n+1)*bz]]
						w_list = [load_img(os.path.join(opt.indir, item))[2] for item in img_list[n*bz:(n+1)*bz]]
						init_image = torch.cat(init_image, dim=0)
						init_image_224 = torch.cat(init_image_224, dim=0)
					# init_image = init_image_list[n].cuda()
					# init_image_224 = init_image_list_224[n].cuda()
					# item = img_list_ori[n]
					# init_image =  load_img(os.path.join(opt.indir, item)).to(device)
					# init_image_224 = load_img_224(os.path.join(opt.indir, item)).to(device)
					init_latent_generator, enc_fea_lq = vq_model.encode(init_image)
					init_latent = model.get_first_stage_encoding(init_latent_generator)
					# init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
					text_init = ['']*init_image.size(0)
					# print(init_image.shape)
					# print(init_image_224.shape)
					img_fea, de_probs = model.classifier(init_image_224)
					img_fea = model.image_proj(img_fea)
					semantic_c = model.get_condtion_prompt(text_init,de_probs,img_fea)

					
					# z_enc = sampler.stochastic_encode(init_latent, torch.tensor([(t_enc-1)*init_image.size(0)]).to(device))
					z_enc = torch.randn(init_latent.shape,device=init_latent.device)

					# noise = torch.randn_like(init_latent)
					# If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
					# t = repeat(torch.tensor([499]), '1 -> b', b=init_image.size(0))
					# t = t.to(device).long()
					# x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
					# x_T = None
                    
					# samples, _ = model.sample(cond=semantic_c, struct_cond=init_latent, batch_size=init_image.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True)
					# x_samples = model.decode_first_stage(samples)
					x_samples_npy = sampler.decode(de_probs,z_enc,semantic_c,init_latent,t_enc,1,semantic_c) 
					# x_samples_npy = x_samples_npy.float()
					# enc_fea_lq =[item.float() for item in enc_fea_lq]
					# de_probs = de_probs.float()
					# print(x_samples_npy)
					x_samples = vq_model.decode(x_samples_npy * 1. / model.scale_factor, enc_fea_lq, de_probs)
					# print(x_samples)
					# x_samples = model.decode_first_stage(x_samples_npy)
					x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

					for i in range(init_image.size(0)):
						img_name = img_list[n*bz+i]
						basename = os.path.splitext(os.path.basename(img_name))[0]
						# x_sample_npy = x_samples_npy[i].cpu().numpy()
						# print(x_sample_npy.shape)
						# np.save(os.path.join(outpath_npy, basename+'.npy'),x_sample_npy)
						x_sample =  255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
						x_sample = x_sample[:h_list[i],:w_list[i],:]
						Image.fromarray(x_sample.astype(np.uint8)).save(
							os.path.join(outpath_sample, basename+'.png'))
							
					# x_samples = model.decode_first_stage(x_samples)
					# x_samples = vq_model.decode(x_samples * 1. / model.scale_factor, enc_fea_lq)

					# if opt.colorfix_type == 'adain':
					# 	x_samples = adaptive_instance_normalization(x_samples, init_image)
					# elif opt.colorfix_type == 'wavelet':
					# 	x_samples = wavelet_reconstruction(x_samples, init_image)
					# x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

					
						# x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')


						# Image.fromarray(x_sample.astype(np.uint8)).save(
						# 	os.path.join(outpath, basename+'.png'))

				toc = time.time()

	print(f"Your samples are ready and waiting for you here: \n{outpath_sample} \n"
		  f" \nEnjoy.")


if __name__ == "__main__":
	main()
	# print(space_timesteps(1000,'ddim200'))
