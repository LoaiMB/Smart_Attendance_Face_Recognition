from models import SiameseNetwork
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import PIL.Image as Image
import os
import torch
from collections import defaultdict
from torchvision.transforms import ToTensor, Normalize, Compose
import cv2

class InferenceDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_to_images = {}

        for label in os.listdir(folder_path):
            label_path = os.path.join(folder_path, label)
            if os.path.isdir(label_path):
                self.label_to_images[label] = [os.path.join(label_path, img) for img in os.listdir(label_path)]
                for img in os.listdir(label_path):
                    self.image_paths.append(os.path.join(label_path, img))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = os.path.basename(os.path.dirname(img_path))
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# Define the transform
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to your inference dataset folder
inference_folder_path = 'inference/'

# Create the dataset and dataloader
inference_dataset = InferenceDataset(inference_folder_path, transform=data_transform)
inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

def capture_image():
    # Capture image from webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    # Save the image
    filename = "inference/image.jpg"
    cv2.imwrite(filename, frame)
    return filename

def run_inference(model, device, transform):
    # Capture image from webcam
    filename = capture_image()
    anchor_img = Image.open(filename).convert('RGB')
    anchor_img = transform(anchor_img).unsqueeze(0).to(device)
    
    votes = defaultdict(int)

    # Run inference on the inference set
    model.eval()
    with torch.no_grad():
        for img, label in inference_loader:
            img = img.to(device)
            output = model(anchor_img, img).squeeze().item()
            
            if output > 0.5:  # Threshold for considering a match
                votes[label[0]] += 1

    # Determine the best match using voting mechanism
    if votes:
        best_match = max(votes, key=votes.get)
        print(f"Best match: {best_match} with {votes[best_match]} votes")
    else:
        print("No match found")

# Example usage
transform = Compose([
    transforms.Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
model = SiameseNetwork()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(torch.load("weights/best_model.pth"))
run_inference(model, device, transform)
