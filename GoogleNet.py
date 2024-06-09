from fastai.vision.all import *
from torchvision.models import googlenet
from pathlib import Path
from fastai.metrics import error_rate


# Path to your dataset
path = Path('/Users/visheshgoyal/Python Projects/Leaf Project/MangoLeafDBAryaShah')

# Load the data
dls = ImageDataLoaders.from_folder(
    path,
    valid_pct = 0.2,
    item_tfms=Resize(448),
    batch_tfms=aug_transforms(mult=2)
)

# Load the GoogLeNet model
googlenet_model = googlenet(pretrained=True, aux_logits=True)

# Check if the model is created correctly
if googlenet_model is None:
    raise ValueError("GoogLeNet model is not instantiated correctly.")

# Remove the auxiliary outputs to fit fastai's expectation of a single output
googlenet_model.aux_logits = False

# Create a CNN learner
learn = Learner(dls, googlenet_model, metrics=error_rate)

# Check if the learner is created correctly
if learn is None:
    raise ValueError("Learner is not created correctly.")

# Train the model
learn.fine_tune(5)

# Optional: Export the learner for inference later
learn.export('/Users/visheshgoyal/Python Projects/Leaf Project/Mangoleaf/Model5/Trained Model/GoogleNetFinal.pkl')

# Display sample predictions
learn.show_results()

# # Load Model
# googleNetModel = load_learner('/Users/visheshgoyal/Python Projects/Leaf Project/Mangoleaf/Model5/Trained Model/GoogleNetFinal.pkl')

# # Load image and test
# img = PILImage.create('')
# pred, pred_idx, probs = googleNetModel.predict(img)

# print(f'Prediction: {pred}, Probability: {probs[pred_idx]:.4f}')