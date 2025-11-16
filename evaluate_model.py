import torch  # Importing the PyTorch library
import torch.nn as nn  # Importing the neural network module from PyTorch
from torch.utils.data import DataLoader  # Importing DataLoader for loading datasets
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import seaborn as sns  # Importing seaborn for enhanced visualizations
from sklearn.metrics import (  # Importing various metrics for evaluation
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score
)
import numpy as np  # Importing NumPy for numerical operations
from train import BreastCancerDataset, get_model, val_transform  # Importing custom dataset and model functions
import pandas as pd  # Importing pandas for data manipulation
from tqdm import tqdm  # Importing tqdm for progress bars

def plot_training_history():  # Function to plot training history
    """Plot training and validation metrics from saved history"""
    try:
        # Try loading from checkpoint first
        checkpoint = torch.load('best_model.pth', map_location='cpu')  # Load the best model checkpoint
        history = checkpoint['history']  # Extract training history from checkpoint
    except:
        # Fall back to separate history file
        history = np.load('training_history.npy', allow_pickle=True).item()  # Load history from a separate file
    
    plt.figure(figsize=(15, 5))  # Set figure size for plots
    
    # Plot training curves
    plt.subplot(1, 2, 1)  # Create a subplot for loss
    plt.plot(history['train_loss'], label='Train Loss')  # Plot training loss
    plt.plot(history['val_loss'], label='Val Loss')  # Plot validation loss
    plt.title('Training and Validation Loss')  # Set title for loss plot
    plt.xlabel('Epoch')  # Set x-axis label
    plt.ylabel('Loss')  # Set y-axis label
    plt.legend()  # Show legend
    plt.grid(True)  # Enable grid
    
    plt.subplot(1, 2, 2)  # Create a subplot for accuracy
    plt.plot(history['train_acc'], label='Train Acc')  # Plot training accuracy
    plt.plot(history['val_acc'], label='Val Acc')  # Plot validation accuracy
    plt.title('Training and Validation Accuracy')  # Set title for accuracy plot
    plt.xlabel('Epoch')  # Set x-axis label
    plt.ylabel('Accuracy')  # Set y-axis label
    plt.legend()  # Show legend
    plt.grid(True)  # Enable grid
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')  # Save the plot as an image
    plt.close()  # Close the plot

def plot_confusion_matrix(y_true, y_pred, classes):  # Function to plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)  # Compute confusion matrix
    plt.figure(figsize=(10, 8))  # Set figure size
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Convert counts to percentages
    
    # Plot with both count and percentage
    sns.heatmap(cm, annot=np.asarray([  # Create a heatmap for confusion matrix
        [f'{count}\n{percent:.1f}%' for count, percent in zip(row_counts, row_percents)]  # Annotate with counts and percentages
        for row_counts, row_percents in zip(cm, cm_percent)
    ]), fmt='', cmap='Blues', xticklabels=classes, yticklabels=classes)  # Set color map and labels
    
    plt.title('Confusion Matrix\n(count and percentage)')  # Set title for confusion matrix
    plt.ylabel('True Label')  # Set y-axis label
    plt.xlabel('Predicted Label')  # Set x-axis label
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')  # Save the confusion matrix plot
    plt.close()  # Close the plot
    
    # Calculate per-class metrics
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)  # Calculate accuracy for each class
    print("\nPer-class Accuracy:")  # Print header for per-class accuracy
    for i, acc in enumerate(per_class_accuracy):  # Iterate over class accuracies
        print(f"{classes[i]}: {acc:.4f} ({acc*100:.1f}%)")  # Print accuracy for each class

def plot_roc_curves(y_true, y_scores, classes):  # Function to plot ROC curves
    plt.figure(figsize=(10, 8))  # Set figure size
    
    # One-hot encode true labels
    y_true_onehot = np.eye(len(classes))[y_true]  # Convert true labels to one-hot encoding
    
    # Plot ROC curve for each class
    colors = ['blue', 'red', 'green']  # Define colors for each class
    for i, (class_name, color) in enumerate(zip(classes, colors)):  # Iterate over classes and colors
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_scores[:, i])  # Compute ROC curve
        roc_auc = auc(fpr, tpr)  # Compute AUC
        plt.plot(fpr, tpr, color=color, lw=2,  # Plot ROC curve
                label=f'{class_name} (AUC = {roc_auc:.2f})')  # Label with AUC
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Plot diagonal line
    plt.xlim([0.0, 1.0])  # Set x-axis limits
    plt.ylim([0.0, 1.05])  # Set y-axis limits
    plt.xlabel('False Positive Rate')  # Set x-axis label
    plt.ylabel('True Positive Rate')  # Set y-axis label
    plt.title('Receiver Operating Characteristic (ROC) Curves')  # Set title for ROC curves
    plt.legend(loc="lower right")  # Show legend
    plt.grid(True)  # Enable grid
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')  # Save the ROC curves plot
    plt.close()  # Close the plot

def plot_precision_recall_curves(y_true, y_scores, classes):  # Function to plot Precision-Recall curves
    plt.figure(figsize=(10, 8))  # Set figure size
    
    # One-hot encode true labels
    y_true_onehot = np.eye(len(classes))[y_true]  # Convert true labels to one-hot encoding
    
    # Plot PR curve for each class
    colors = ['blue', 'red', 'green']  # Define colors for each class
    for i, (class_name, color) in enumerate(zip(classes, colors)):  # Iterate over classes and colors
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_scores[:, i])  # Compute precision-recall curve
        ap = average_precision_score(y_true_onehot[:, i], y_scores[:, i])  # Compute average precision
        plt.plot(recall, precision, color=color, lw=2,  # Plot PR curve
                label=f'{class_name} (AP = {ap:.2f})')  # Label with average precision
    
    plt.xlabel('Recall')  # Set x-axis label
    plt.ylabel('Precision')  # Set y-axis label
    plt.title('Precision-Recall Curves')  # Set title for PR curves
    plt.legend(loc="lower left")  # Show legend
    plt.grid(True)  # Enable grid
    plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')  # Save the PR curves plot
    plt.close()  # Close the plot

def evaluate_model(model, data_loader, device, classes):  # Function to evaluate the model
    model.eval()  # Set model to evaluation mode
    all_preds = []  # List to store all predictions
    all_labels = []  # List to store all true labels
    all_scores = []  # List to store all scores
    
    print("\nRunning evaluation...")  # Print evaluation start message
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):  # Loop over data loader
            inputs = inputs.to(device)  # Move inputs to device
            labels = labels.to(device)  # Move labels to device
            
            outputs = model(inputs)  # Get model outputs
            scores = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            _, preds = torch.max(outputs, 1)  # Get predicted classes
            
            all_preds.extend(preds.cpu().numpy())  # Store predictions
            all_labels.extend(labels.cpu().numpy())  # Store true labels
            all_scores.extend(scores.cpu().numpy())  # Store scores
    
    all_preds = np.array(all_preds)  # Convert predictions to NumPy array
    all_labels = np.array(all_labels)  # Convert true labels to NumPy array
    all_scores = np.array(all_scores)  # Convert scores to NumPy array
    
    # Plot training history
    print("\nGenerating training history plot...")  # Print message for training history plot
    plot_training_history()  # Call function to plot training history
    
    # Generate and save confusion matrix
    print("\nGenerating confusion matrix...")  # Print message for confusion matrix
    plot_confusion_matrix(all_labels, all_preds, classes)  # Call function to plot confusion matrix
    
    # Generate and save ROC curves
    print("\nGenerating ROC curves...")  # Print message for ROC curves
    plot_roc_curves(all_labels, all_scores, classes)  # Call function to plot ROC curves
    
    # Generate and save Precision-Recall curves
    print("\nGenerating Precision-Recall curves...")  # Print message for PR curves
    plot_precision_recall_curves(all_labels, all_scores, classes)  # Call function to plot PR curves
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=classes)  # Generate classification report
    with open('classification_report.txt', 'w') as f:  # Open file to save report
        f.write(report)  # Write report to file
    
    # Calculate overall accuracy
    accuracy = (all_preds == all_labels).mean()  # Calculate overall accuracy
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")  # Print overall accuracy
    
    return report  # Return classification report

def main():  # Main function
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device to GPU if available
    print(f"Using device: {device}")  # Print the device being used
    
    # Dataset path
    data_dir = "D:/gaurav/Breast Cancer_DB24/Breast Cancer/Breast Cancer/Dataset"  # Path to dataset
    
    # Classes
    classes = ['benign', 'malignant', 'normal']  # Define class labels
    
    # Create dataset and data loader
    print("\nLoading dataset...")  # Print message for loading dataset
    dataset = BreastCancerDataset(data_dir, transform=val_transform)  # Create dataset
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)  # Create data loader
    
    # Load the trained model
    print("\nLoading model...")  # Print message for loading model
    model = get_model(num_classes=3)  # Get the model
    checkpoint = torch.load('best_model.pth', map_location=device)  # Load the best model checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])  # Load model state
    model = model.to(device)  # Move model to device
    
    best_acc = checkpoint['best_acc']  # Get best accuracy from checkpoint
    print(f"Best validation accuracy from training: {best_acc:.4f} ({best_acc*100:.1f}%)")  # Print best accuracy
    
    # Evaluate the model
    report = evaluate_model(model, data_loader, device, classes)  # Call function to evaluate model
    print("\nClassification Report:")  # Print classification report header
    print(report)  # Print classification report
    
    print("\nEvaluation completed. Check the following files:")  # Print completion message
    print("1. confusion_matrix.png - Shows prediction distribution")  # File for confusion matrix
    print("2. roc_curves.png - Shows true/false positive trade-off")  # File for ROC curves
    print("3. precision_recall_curves.png - Shows precision/recall trade-off")  # File for PR curves
    print("4. training_history.png - Shows training progression")  # File for training history
    print("5. classification_report.txt - Detailed metrics")  # File for classification report

if __name__ == "__main__":  # Check if the script is being run directly
    main()  # Call the main function