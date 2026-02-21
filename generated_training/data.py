"""data.py — HuggingFace dataset loader for TinyStories"""
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T

DATASET_ID  = "roneneldan/TinyStories"
BATCH_SIZE  = 32
MAX_LENGTH  = 512   # tokens

def get_dataloaders(batch_size=BATCH_SIZE, hf_token=None):
    ds = load_dataset(DATASET_ID, token=hf_token)
    # TODO: define transforms and adjust column names for your dataset
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    train_ds = ds["train"]
    val_ds   = ds.get("validation") or ds.get("test")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size) if val_ds else None,
    )
