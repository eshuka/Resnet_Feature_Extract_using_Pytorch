import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
from PIL import Image
import torch.nn as nn
import torch.backends.cudnn as cudnn

#f1 = open('test1.txt','w')
#f2 = open('test2.txt','w')

model_file =  #model file
file_name = #label file
img_name = #test image

model = torch.load(model_file)

centre_crop = trn.Compose([
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][0:])
classes = tuple(classes)

img = Image.open(img_name)
img = img.convert('RGB')
img = img.resize((256, 256), Image.ANTIALIAS)

use_gpu = torch.cuda.is_available()

input_img = V(centre_crop(img).unsqueeze(0), volatile=True)

if use_gpu:
    input_img = input_img.cuda()

feature_map = list(model.module.children())
#f1.write(str(feature_map))
feature_map.pop()
#f2.write(str(feature_map))
extractor = nn.Sequential(*feature_map)

if use_gpu:
    model.cuda()
    extractor.cuda()
    cudnn.benchmark = True

extractor.eval()

feature = extractor(input_img)

print(feature.size())