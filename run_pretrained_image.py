import torchvision.models as models
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

# --------- Preprocess function ----------
def preprocess():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# --------- Read ImageNet classes ----------
def read_classes():
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories

# --------- Setup ---------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(device).eval()
categories = read_classes()
transform = preprocess()

# --------- Load image instead of webcam ---------
image_path = "pic1.jpg"
frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError(f"Could not load {image_path}")

# Convert and preprocess
input_tensor = transform(frame)
input_batch = input_tensor.unsqueeze(0).to(device)

# --------- Inference ---------
with torch.no_grad():
    output = model(input_batch)
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# --------- Top 5 predictions ---------
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    label = categories[top5_catid[i]]
    prob = top5_prob[i].item()
    cv2.putText(frame, f"{prob*100:.2f}% {label}",
                (15, (i+1)*30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2, cv2.LINE_AA)
    print(f"{label}: {prob*100:.2f}%")

# --------- Show/Save ---------
cv2.imshow("Classification Result", frame)
# cv2.imwrite("classified_pic1.jpg", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()