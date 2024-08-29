### Train
```
cd classification
python3 train.py
python3 main.py --train --base configs/MPerceiver/all_in_one.yaml --gpus 0,1,2,3,4,5,6,7, --name all_in_one --scale_lr False
```

### Inference
```
cd classification
python3 clip_classify_eval.py
python3 scripts/ddim_sample_vae.py --config configs/stableSRNew/all_in_one.yaml --ckpt CKPT_PATH --vqgan_ckpt VQGANCKPT_PATH --indir INPUT_PATH --outdir OUT_DIR --ddim_steps 50
```
