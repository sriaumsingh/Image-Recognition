import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import your model architecture
from run1 import ASLNet  

def load_model(model_path):
    model = ASLNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return transform(image).unsqueeze(0)

def get_prediction(model, image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

def main():
    model_path = 'asl_recognition_model.pth'
    model = load_model(model_path)
    
    # Define the classes (0-9 and A-Z)
    classes = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z')+1)]
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        processed_frame = preprocess_image(frame)
        
        # Get prediction
        prediction = get_prediction(model, processed_frame)
        predicted_class = classes[prediction]
        
        # Draw the prediction on the frame
        cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('ASL Recognition', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()