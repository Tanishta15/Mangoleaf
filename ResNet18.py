from fastai.vision.all import *
from pathlib import Path
from sklearn.metrics import *

# Path to your dataset
path = Path('/Users/visheshgoyal/Python Projects/Leaf Project/MangoLeafDBAryaShah')

# Load the data
dls = ImageDataLoaders.from_folder(
    path,
    valid_pct = 0.2,
    item_tfms=Resize(448),
    batch_tfms=aug_transforms(mult=2)
)

# # Create a CNN learner
# learn = cnn_learner(dls, resnet18, metrics=error_rate)

# # Train the model
# learn.fine_tune(5)

# # Export the learner for inference later
export_path = Path('//Users/visheshgoyal/Python Projects/Leaf Project/Mangoleaf/Model3/Trained Model/Resnet18Final.pkl')
# learn.export(export_path)

# Load the exported learner
learn_inf = load_learner(export_path)

# # Predict on a new image
# img_path = Path('/path/to/new/image.jpg')
# img = PILImage.create(img_path)
# pred, pred_idx, probs = learn_inf.predict(img)
# print(f'Prediction: {pred}, Probability: {probs[pred_idx]:.4f}')

# # Calculating metrics
# preds, targs = learn_inf.get_preds(dl=dls.valid)
# pred_labels = preds.argmax(dim=1)
# pred_labels = pred_labels.numpy()
# targs = targs.numpy()
# f1 = f1_score(targs, pred_labels, average='macro')
# print(f"F1 Score: {f1}")