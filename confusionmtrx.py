import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from fastai.vision.all import *

# Load the trained model
learn = load_learner('path_to_your_exported_model.pkl')

# Load the test dataset
test_items = get_image_files('path_to_your_test_images')

# Create a DataLoader for the test dataset
test_dl = learn.dls.test_dl(test_items)

# Get predictions
preds, y_true = learn.get_preds(dl=test_dl)
preds_class = preds.argmax(dim=1)

# Convert predicted class labels and ground truth to numpy arrays
preds_np = preds_class.numpy()
y_true_np = y_true.numpy()

# Define the class labels
class_labels = ["Healthy","Infected"] 

# Generate confusion matrix
cm = confusion_matrix(y_true_np, preds_np, labels=np.arange(len(class_labels)))

# Convert confusion matrix to a DataFrame for visualization
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
