import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# 이미지 경로 설정 및 로드
image_path = "capture.JPG"
image = Image.open(image_path).convert('RGB')

# 모델 로드
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

# 이미지 전처리
preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image).unsqueeze(0)

# 결과 시각화
plt.imshow(image)
plt.show()
