import torch, cv2
from torch.autograd import Variable as V
from torchvision import transforms as trn
from PIL import Image
import torch.nn as nn
import torch.backends.cudnn as cudnn

#f1 = open('test1.txt','w')
#f2 = open('test2.txt','w')

model_file = 'models/resnet-50.pth.tar'
model = torch.load(model_file)
centre_crop = trn.Compose([
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

file_name = 'txt/categories.txt'

classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][0:])
classes = tuple(classes)

img_name = 'images/play.jpeg'

img = Image.open(img_name)
cv2_image = cv2.imread(img_name)
img = img.convert('RGB')
img = img.resize((224, 224), Image.ANTIALIAS)

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

sample_out = extractor(input_img)

print(sample_out.size())