from dataclasses import dataclass

import clip
import torch

from PIL import Image
from typing import List


@dataclass
class StoredImage:
    path: str
    embedding: List


class ClipWrapper:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def preprocess_image(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        return image

    def create_image_embedding(self, image_path):
        preprocessed_image = self.preprocess_image(image_path)
        image_embedding = self.model.encode_image(preprocessed_image)[0].cpu().detach().numpy()
        list_image_embedding = [float(i) for i in image_embedding]  # Elastic search only accepts JSON valid fields

        return StoredImage(image_path, list_image_embedding)

    def create_text_embedding(self, text):
        tokenized_text = clip.tokenize([text]).to(self.device)
        text_embedding = self.model.encode_text(tokenized_text)[0].cpu().detach().numpy()
        list_text_embedding = [float(i) for i in text_embedding]  # Elastic search only accepts JSON valid fields

        return list_text_embedding
