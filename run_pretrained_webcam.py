import torchvision.models as models
import torch
import time
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

# download the image classes from https://raw.githubusercontent.com/Lasagne/Recipes/master/examples/resnet50/imagenet_classes.txt
# and put in the same folder as this python file

def preprocess():
    """
    Define the transform for the input image/frames.
    Resize, crop, convert to tensor, and apply ImageNet normalization stats.
    """
    transform =  transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),])
    return transform


def read_classes():
    """
    Load the ImageNet class names.
    """
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories


# Set the computation device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Initialize the model.
# model = torch.load("full_model.pth")
model = models.resnet50(pretrained=True)
model.eval()
model.to(device)
# Load the ImageNet class names.
categories = read_classes()
# Initialize the image transforms.
transform = preprocess()


cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read() 
    
#     rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Apply transforms to the input image.
    input_tensor = transform(frame)
    # Add the batch dimension.
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)
    
    with torch.no_grad():
        start_time = time.time()
        output = model(input_batch)
        end_time = time.time()
    # Get the softmax probabilities.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Check the top 5 categories that are predicted.
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        cv2.putText(frame, f"{top5_prob[i].item()*100:.3f}%", (15, (i+1)*30), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{categories[top5_catid[i]]}", (160, (i+1)*30), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        print(categories[top5_catid[i]], top5_prob[i].item())
    
    
    cv2.imshow('Raw Webcam Feed', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        break

cv2.destroyAllWindows()
