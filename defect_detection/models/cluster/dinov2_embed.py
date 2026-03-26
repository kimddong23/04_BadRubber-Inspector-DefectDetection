import torch
import torchvision.transforms as T
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Model
# -----------------------------
_model = None

def load_model(model_name="dinov2_vitb14"):
    global _model
    if _model is None:
        _model = torch.hub.load("facebookresearch/dinov2", model_name)
        _model.eval().to(device)
    return _model


# -----------------------------
# Transform
# -----------------------------
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    )
])


# -----------------------------
# Single image embedding
# -----------------------------
def get_embedding(image):

    model = load_model()

    img = Image.fromarray(image).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        feat = model(img)

    feat = feat.squeeze()

    feat = torch.nn.functional.normalize(feat, dim=-1)

    return feat

# -----------------------------
# Batch embedding
# -----------------------------
def get_embeddings_batch(images):
    model = load_model()

    imgs = []
    for img_np in images:
        img = Image.fromarray(img_np).convert("RGB")
        img = transform(img)
        imgs.append(img)

    imgs = torch.stack(imgs).to(device, non_blocking=True)

    with torch.inference_mode():
        feats = model(imgs)

    feats = torch.nn.functional.normalize(feats, dim=1)

    return feats