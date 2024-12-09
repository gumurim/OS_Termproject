import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


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

# GPU 설정 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
input_tensor = input_tensor.to(device)

# 예측 수행
with torch.no_grad():
    output = model(input_tensor)['out'][0]
output_predictions = output.argmax(0)

# 마스킹
person_class = 15
mask = output_predictions == person_class
image_np = np.array(image)
image_np[mask.cpu().numpy()] = [255, 0, 0]


# 결과 시각화 
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Segmented Image")
plt.imshow(image_np)
plt.axis('off')

plt.show()
