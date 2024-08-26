import torch
from PIL import Image
import open_clip
import os 
from tqdm import tqdm 

#Test CLIP's zero-shot classification of different degradation types
model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
model = model.cuda()
tokenizer = open_clip.get_tokenizer('ViT-H-14')
text = tokenizer(["a photo with rain", "a photo with snow", "a photo with haze"])
text = text.cuda()

path = '/input/'
files = os.listdir(path)
num = len(files)
accuracte = 0
for file in tqdm(files):
    image = preprocess(Image.open(path+file)).unsqueeze(0)
    image = image.cuda()
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    if text_probs[0].argmax() == 2:
        accuracte += 1

print(accuracte/num)
