import os
import torch.nn as nn
import torch
from torch import Tensor
from typing import Dict, List
import torch.nn.functional as F


class CLIP_wrapper(nn.Module):
    def __init__(self, modelname: str = "ViT-B/32", device: str = "cpu"):
        super().__init__()
        self.device = device

        import clip

        model, preprocess = clip.load(modelname, device)
        self.tokenizer = clip.tokenize
        self.clip_model = model.eval()

        # Freeze the weights just in case
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> nn.Module:
        # override it to be always false
        self.training = False
        for module in self.children():
            module.train(False)
        return self

    @torch.no_grad()
    def forward(self, texts: List[str], device=None) -> Dict:
        device = device if device is not None else self.device
        tokens = self.tokenizer(texts, truncate=True).to(device)
        return self.clip_model.encode_text(tokens).float()


class HF_wrapper(nn.Module):
    def __init__(
        self, modelpath: str, mean_pooling: bool = False, device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = device

        from transformers import AutoTokenizer, AutoModel, T5EncoderModel
        from transformers import logging

        logging.set_verbosity_error()

        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)

        if modelpath == "google/flan-t5-xl":
            # only load the encoder not the decoder as well
            self.text_model = T5EncoderModel.from_pretrained(modelpath)
        else:
            # Text model
            self.text_model = AutoModel.from_pretrained(modelpath)
        # Then configure the model
        self.text_encoded_dim = self.text_model.config.hidden_size

        if mean_pooling:
            self.forward = self.forward_pooling

        # put it in eval mode by default
        self.eval()

        # Freeze the weights just in case
        for param in self.parameters():
            param.requires_grad = False

        self.to(device)

    def train(self, mode: bool = True) -> nn.Module:
        # override it to be always false
        self.training = False
        for module in self.children():
            module.train(False)
        return self

    @torch.no_grad()
    def forward(self, texts: List[str], device=None) -> Dict:
        device = device if device is not None else self.device

        squeeze = False
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True

        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model.encoder(**encoded_inputs.to(device))
        length = encoded_inputs.attention_mask.to(dtype=bool).sum(1)

        if squeeze:
            x_dict = {"x": output.last_hidden_state.detach()[0], "length": length[0]}
        else:
            x_dict = {"x": output.last_hidden_state.detach(), "length": length}
        return x_dict

    @torch.no_grad()
    def forward_pooling(self, texts: List[str], device=None) -> Tensor:
        device = device if device is not None else self.device

        squeeze = False
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True

        # From: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded_inputs.to(device))
        attention_mask = encoded_inputs["attention_mask"]

        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = output["last_hidden_state"]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        if squeeze:
            sentence_embeddings = sentence_embeddings[0]
        return sentence_embeddings


def TextToEmb(modelpath: str, mean_pooling: bool = False, device: str = "cpu"):
    if modelpath == "clip":
        modelpath = "ViT-B/32"

    # clip models
    if modelpath in [
        "RN50",
        "RN101",
        "RN50x4",
        "RN50x16",
        "RN50x64",
        "ViT-B/32",
        "ViT-B/16",
        "ViT-L/14",
        "ViT-L/14@336px",
    ]:
        return CLIP_wrapper(modelpath, device=device)
    # hugging face
    else:
        return HF_wrapper(modelpath, mean_pooling, device)
