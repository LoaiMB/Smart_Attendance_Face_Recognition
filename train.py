from models import SiameseNetwork
from data import build_loader, train_transform, val_transform
import torch
import torch.nn as nn
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np
import datetime


# Load the LFW dataset
lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
X = lfw_people.images
y = lfw_people.target

# Split into train and test and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create data loaders
train_loader = build_loader(X_train, y_train, transform=train_transform) 
val_loader =  build_loader(X_val, y_val, transform=val_transform)

test_loader = build_loader(X_test, y_test, transform=val_transform)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train = False
model = SiameseNetwork().to(device)
if train:
  criterion = nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  num_epochs = 5
  best_loss = float('inf')
  for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for img1, img2, label in train_loader:
          img1, img2, label = img1.to(device), img2.to(device), label.to(device).float()
          
          output = model(img1, img2).squeeze()
          loss = criterion(output, label)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          running_loss += loss.item()

          # Save the best model
          if epoch == 0 or loss.item() < best_loss:
              best_loss = loss.item()
              model_save_path = f"weights/siamese_resnet_18_{datetime.now().strftime('%Y%m%d%H%M')}.pth"
              torch.save(model.state_dict(), model_save_path)

      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

      # Validation
      model.eval()
      val_loss = 0.0
      with torch.no_grad():
          for img1, img2, label in val_loader:
              img1, img2, label = img1.to(device), img2.to(device), label.to(device).float()
              output = model(img1, img2).squeeze()
              loss = criterion(output, label)
              val_loss += loss.item()

      print(f'Validation Loss: {val_loss/len(val_loader)}')
else:
  model.load_state_dict(torch.load("weights/best_model.pth"))
  model.eval()  # Set the model to evaluation mode

evaluate = True
visualize = True

if evaluate:
    from utils import evaluate_model
    evaluate_model(model, test_loader, device)
if visualize:
    from utils import visualize_predictions
    visualize_predictions(model, test_loader, device)