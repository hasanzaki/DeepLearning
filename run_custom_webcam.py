import cv2
import torch
import torchvision.transforms as transforms

# load the model
model = torch.load("spectacle.pt", map_location=torch.device("cpu"))
model.eval()
model.to('cpu')

class_labels = ['nospec', 'spec']

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])

# reading the image
capture = cv2.VideoCapture(1)

while True:

    isTrue, frame = capture.read()
    
    image = frame.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    image = image.unsqueeze(0) # add batch dimension

    # feedforward/ inference
    with torch.no_grad():
        output = model(image)

    # postprocess output/ label
    _,predicted_class = output.max(1)
    predicted_class = predicted_class.item()

    predicted_class_name = class_labels[predicted_class]
    print(predicted_class_name)

    # label = f"Class: {predicted_class}"
    cv2.putText(frame,predicted_class_name,(10,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)

    cv2.imshow('myVideo', frame)

    cv2.waitKey(1)


capture.release()
out.release()

cv2.destroyAllWindows()

