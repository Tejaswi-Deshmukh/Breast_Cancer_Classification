

import torch  # Importing the PyTorch library
import torch.nn as nn  # Importing the neural network module from PyTorch
import torch.optim as optim  # Importing the optimization module from PyTorch
from torch.utils.data import Dataset, DataLoader, Subset  # Importing dataset utilities
from torchvision import transforms, models  # Importing transforms and models from torchvision
from PIL import Image  # Importing the Python Imaging Library for image processing
import os  # Importing the os module for file and directory operations
from tqdm import tqdm  # Importing tqdm for progress bars
import numpy as np  # Importing NumPy for numerical operations
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Importing learning rate scheduler
from torch.utils.data import WeightedRandomSampler  # Importing weighted random sampler
import albumentations as A  # Importing albumentations for image augmentation
from albumentations.pytorch import ToTensorV2  # Importing ToTensorV2 for converting images to tensors
from sklearn.model_selection import train_test_split  # Importing train_test_split for dataset splitting

# Enhanced data augmentation
train_transform = A.Compose([  # Composing a series of augmentations for training data
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),  # Randomly crop and resize images
    A.HorizontalFlip(p=0.5),  # Randomly flip images horizontally
    A.VerticalFlip(p=0.5),  # Randomly flip images vertically
    A.RandomRotate90(p=0.5),  # Randomly rotate images by 90 degrees
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),  # Randomly shift, scale, and rotate images
    A.OneOf([  # Apply one of the following augmentations
        A.GaussNoise(p=1),  # Add Gaussian noise
        A.GaussianBlur(p=1),  # Apply Gaussian blur
        A.MotionBlur(p=1),  # Apply motion blur
    ], p=0.3),  # Probability of applying the above augmentations
    A.OneOf([  # Apply one of the following distortions
        A.OpticalDistortion(p=1),  # Apply optical distortion
        A.GridDistortion(p=1),  # Apply grid distortion
        A.ElasticTransform(p=1),  # Apply elastic transformation
    ], p=0.3),  # Probability of applying the above distortions
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),  # Randomly change brightness, contrast, saturation, and hue
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
    ToTensorV2(),  # Convert images to PyTorch tensors
])

val_transform = A.Compose([  # Composing a series of augmentations for validation data
    A.Resize(224, 224),  # Resize images to 224x224
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
    ToTensorV2(),  # Convert images to PyTorch tensors
])

class BreastCancerDataset(Dataset):  # Defining a custom dataset class
    def __init__(self, data_dir, transform=None, is_train=True):  # Initializing the dataset
        self.data_dir = data_dir  # Directory containing the dataset
        self.transform = transform  # Transformations to apply
        self.is_train = is_train  # Flag indicating if this is the training set
        self.classes = ['benign', 'malignant', 'normal']  # Class labels
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # Mapping class names to indices
        
        self.images = []  # List to store image file paths
        self.labels = []  # List to store corresponding labels
        
        # Load data and compute class weights
        class_counts = [0] * len(self.classes)  # Initialize class counts
        for class_name in self.classes:  # Iterate over each class
            class_dir = os.path.join(data_dir, class_name)  # Get the directory for the class
            if os.path.exists(class_dir):  # Check if the directory exists
                for img_name in os.listdir(class_dir):  # Iterate over images in the class directory
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):  # Check for valid image file extensions
                        self.images.append(os.path.join(class_dir, img_name))  # Append image path
                        label = self.class_to_idx[class_name]  # Get the label index
                        self.labels.append(label)  # Append label
                        class_counts[label] += 1  # Increment class count
        
        # Compute class weights for balanced sampling
        total_samples = sum(class_counts)  # Total number of samples
        self.class_weights = [total_samples / (len(self.classes) * count) for count in class_counts]  # Calculate class weights
        self.sample_weights = [self.class_weights[label] for label in self.labels]  # Calculate sample weights based on labels
        
        print(f"Dataset size: {len(self.images)}")  # Print dataset size
        print(f"Class distribution: {class_counts}")  # Print class distribution
        print(f"Class weights: {self.class_weights}")  # Print class weights
    
    def __len__(self):  # Method to get the length of the dataset
        return len(self.images)  # Return number of images
    
    def __getitem__(self, idx):  # Method to get an item from the dataset
        img_path = self.images[idx]  # Get the image path
        label = self.labels[idx]  # Get the corresponding label
        
        try:
            # Load and convert to RGB
            image = Image.open(img_path).convert('RGB')  # Open and convert image to RGB
            image = np.array(image)  # Convert image to NumPy array
            
            # Apply transforms
            if self.transform:  # Check if a transform is provided
                transformed = self.transform(image=image)  # Apply the transform
                image = transformed['image']  # Get the transformed image
            else:  # If no transform is provided
                # If no transform is provided, at least resize to a consistent size
                transform = A.Compose([  # Compose a default transform
                    A.Resize(224, 224),  # Resize to 224x224
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
                    ToTensorV2(),  # Convert to tensor
                ])
                transformed = transform(image=image)  # Apply the default transform
                image = transformed['image']  # Get the transformed image
            
            return image, label  # Return the image and label
            
        except Exception as e:  # Handle exceptions
            print(f"Error loading image {img_path}: {str(e)}")  # Print error message
            # Return a default image in case of error
            default_image = np.zeros((224, 224, 3), dtype=np.uint8)  # Create a default black image
            if self.transform:  # Check if a transform is provided
                transformed = self.transform(image=default_image)  # Apply the transform
                default_image = transformed['image']  # Get the transformed image
            else:  # If no transform is provided
                transform = A.Compose([  # Compose a default transform
                    A.Resize(224, 224),  # Resize to 224x224
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
                    ToTensorV2(),  # Convert to tensor
                ])
                transformed = transform(image=default_image)  # Apply the default transform
                default_image = transformed['image']  # Get the transformed image
            return default_image, label  # Return the default image and label

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device to GPU if available

# Load pre-trained ResNet model
def get_model(num_classes=3):  # Function to get the model
    # Use EfficientNet-B4 instead of ResNet
    model = models.efficientnet_b4(pretrained=True)  # Load pre-trained EfficientNet-B4 model
    
    # Add dropout for regularization
    model.classifier = nn.Sequential(  # Modify the classifier
        nn.Dropout(p=0.4),  # Add dropout layer
        nn.Linear(model.classifier[1].in_features, 512),  # Add linear layer
        nn.ReLU(),  # Add ReLU activation
        nn.Dropout(p=0.4),  # Add another dropout layer
        nn.Linear(512, num_classes)  # Add final linear layer for output
    )
    return model  # Return the modified model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30):  # Function to train the model
    best_acc = 0.0  # Initialize best accuracy
    patience = 10  # Set patience for early stopping
    patience_counter = 0  # Initialize patience counter
    
    # Training history
    history = {  # Dictionary to store training history
        'train_loss': [], 'train_acc': [],  # Lists for training loss and accuracy
        'val_loss': [], 'val_acc': []  # Lists for validation loss and accuracy
    }
    
    for epoch in range(num_epochs):  # Loop over epochs
        print(f'Epoch {epoch+1}/{num_epochs}')  # Print current epoch
        print('-' * 10)  # Print separator
        
        # Training phase
        model.train()  # Set model to training mode
        running_loss = 0.0  # Initialize running loss
        running_corrects = 0  # Initialize running correct predictions
        
        for inputs, labels in tqdm(train_loader, desc='Training'):  # Loop over training data
            inputs = inputs.to(device)  # Move inputs to device
            labels = labels.to(device)  # Move labels to device
            
            optimizer.zero_grad()  # Zero the gradients
            
            outputs = model(inputs)  # Forward pass
            _, preds = torch.max(outputs, 1)  # Get predictions
            loss = criterion(outputs, labels)  # Calculate loss
            
            loss.backward()  # Backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
            optimizer.step()  # Update weights
            
            running_loss += loss.item() * inputs.size(0)  # Update running loss
            running_corrects += torch.sum(preds == labels.data)  # Update correct predictions
        
        epoch_loss = running_loss / len(train_loader.dataset)  # Calculate average loss for the epoch
        epoch_acc = running_corrects.double() / len(train_loader.dataset)  # Calculate accuracy for the epoch
        
        history['train_loss'].append(epoch_loss)  # Append training loss to history
        history['train_acc'].append(epoch_acc.item())  # Append training accuracy to history
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')  # Print training loss and accuracy
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        running_loss = 0.0  # Initialize running loss for validation
        running_corrects = 0  # Initialize running correct predictions for validation
        
        with torch.no_grad():  # Disable gradient calculation
            for inputs, labels in tqdm(val_loader, desc='Validation'):  # Loop over validation data
                inputs = inputs.to(device)  # Move inputs to device
                labels = labels.to(device)  # Move labels to device
                
                outputs = model(inputs)  # Forward pass
                _, preds = torch.max(outputs, 1)  # Get predictions
                loss = criterion(outputs, labels)  # Calculate loss
                
                running_loss += loss.item() * inputs.size(0)  # Update running loss
                running_corrects += torch.sum(preds == labels.data)  # Update correct predictions
        
        val_loss = running_loss / len(val_loader.dataset)  # Calculate average validation loss
        val_acc = running_corrects.double() / len(val_loader.dataset)  # Calculate validation accuracy
        
        history['val_loss'].append(val_loss)  # Append validation loss to history
        history['val_acc'].append(val_acc.item())  # Append validation accuracy to history
        
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')  # Print validation loss and accuracy
        
        # Learning rate scheduling
        scheduler.step(val_loss)  # Step the scheduler based on validation loss
        
        # Save best model and early stopping
        if val_acc > best_acc:  # Check if current validation accuracy is the best
            best_acc = val_acc  # Update best accuracy
            torch.save({  # Save the model state
                'epoch': epoch,  # Save current epoch
                'model_state_dict': model.state_dict(),  # Save model state
                'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state
                'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
                'best_acc': best_acc,  # Save best accuracy
                'history': history  # Save training history
            }, 'best_model.pth')  # Save to file
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increment patience counter
            
        if patience_counter >= patience:  # Check for early stopping
            print(f'Early stopping triggered after epoch {epoch+1}')  # Print early stopping message
            break  # Exit training loop
            
        print()  # Print a new line
    
    return history  # Return training history

def main():  # Main function
    # Set random seeds for reproducibility
    torch.manual_seed(42)  # Set PyTorch random seed
    np.random.seed(42)  # Set NumPy random seed
    
    # Dataset paths
    data_dir = "D:/gaurav/Breast Cancer_DB24/Breast Cancer/Breast Cancer/Dataset"  # Path to dataset
    
    # Create train and validation datasets with appropriate transforms
    train_dataset = BreastCancerDataset(data_dir, transform=train_transform, is_train=True)  # Create training dataset
    val_dataset = BreastCancerDataset(data_dir, transform=val_transform, is_train=False)  # Create validation dataset
    
    # Create train/val splits using indices
    indices = list(range(len(train_dataset)))  # Create a list of indices
    train_indices, val_indices = train_test_split(  # Split indices into training and validation sets
        indices, test_size=0.2, random_state=42,  # 20% for validation
        stratify=train_dataset.labels  # Stratify based on labels
    )
    
    # Create final datasets
    train_subset = Subset(train_dataset, train_indices)  # Create subset for training
    val_subset = Subset(val_dataset, val_indices)  # Create subset for validation
    
    # Create weighted sampler for training set
    train_labels = [train_dataset.labels[i] for i in train_indices]  # Get labels for training indices
    class_weights = train_dataset.class_weights  # Get class weights
    sample_weights = [class_weights[label] for label in train_labels]  # Calculate sample weights based on labels
    
    sampler = WeightedRandomSampler(  # Create a weighted random sampler
        weights=sample_weights,  # Use calculated sample weights
        num_samples=len(train_subset),  # Number of samples to draw
        replacement=True  # Allow replacement
    )
    
    # Create data loaders
    train_loader = DataLoader(  # Create data loader for training
        train_subset,
        batch_size=16,  # Set batch size
        sampler=sampler,  # Use weighted sampler
        num_workers=4,  # Number of worker threads
        pin_memory=True  # Pin memory for faster data transfer
    )
    
    val_loader = DataLoader(  # Create data loader for validation
        val_subset,
        batch_size=16,  # Set batch size
        shuffle=False,  # Do not shuffle validation data
        num_workers=4,  # Number of worker threads
        pin_memory=True  # Pin memory for faster data transfer
    )
    
    print(f"Training set size: {len(train_subset)}")  # Print size of training set
    print(f"Validation set size: {len(val_subset)}")  # Print size of validation set
    
    # Initialize model, criterion, and optimizer
    model = get_model(num_classes=3)  # Get the model
    model = model.to(device)  # Move model to device
    
    # Use weighted cross entropy loss for class imbalance
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))  # Initialize loss function
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # Initialize optimizer
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(  # Initialize learning rate scheduler
        optimizer, 
        mode='min',  # Reduce learning rate when validation loss plateaus
        factor=0.1,  # Factor by which to reduce the learning rate
        patience=5,  # Number of epochs with no improvement after which learning rate will be reduced
        verbose=True,  # Print messages when learning rate is reduced
        min_lr=1e-6  # Minimum learning rate
    )
    
    # Train the model
    print("Starting training...")  # Print starting message
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30)  # Train the model
    
    # Save training history
    np.save('training_history.npy', history)  # Save training history to file
    print("Training completed!")  # Print completion message

if __name__ == "__main__":  # Check if the script is being run directly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device to GPU if available
    print(f"Using device: {device}")  # Print the device being used
    main()  # Call the main function 