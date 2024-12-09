import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 이미지 로드
image_path = "capture.JPG"
image = Image.open(image_path)

# 모델 로드
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)

# 이미지 전처리
transform = transforms.ToTensor()
image_tensor = transform(image)

# 결과 확인
plt.imshow(image)
plt.show()
