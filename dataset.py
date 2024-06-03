from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomCrop(size=(224, 224), padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SiameseLFWDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.label_to_images = {}
        for i in range(len(data)):
            label = targets[i]
            if label not in self.label_to_images:
                self.label_to_images[label] = []
            self.label_to_images[label].append(i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1, label1 = self.data[idx], self.targets[idx]

        if self.transform:
            img1 = self.transform(img1)

        if random.choice([True, False]):
            # Positive pair
            idx2 = random.choice(self.label_to_images[label1])
            img2, label2 = self.data[idx2], self.targets[idx2]
            label = 1
            # print(f"Positive pair: idx1={idx}, idx2={idx2}, label1={label1}, label2={label2}")
        else:
            # Negative pair
            label2 = random.choice([l for l in self.label_to_images.keys() if l != label1])
            idx2 = random.choice(self.label_to_images[label2])
            img2, label2 = self.data[idx2], self.targets[idx2]
            label = 0
            # print(f"Negative pair: idx1={idx}, idx2={idx2}, label1={label1}, label2={label2}")

        if self.transform:
            img2 = self.transform(img2)

        return img1, img2, label
    
def build_loader(data, label, batch_size, shuffle=True, transform=None):
    dataset = SiameseLFWDataset(data, label, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


