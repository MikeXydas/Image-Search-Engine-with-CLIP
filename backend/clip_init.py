"""
This file is used to download the CLIP model when building the Dockerfile.
"""
import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
