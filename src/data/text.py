import os
import orjson
import json
import torch
import numpy as np
from tqdm import tqdm

from src.model.text_encoder import TextToEmb


class TextEmbeddings:
    name = "text_embeddings"

    def __init__(
        self,
        dataname: str,
        modelname: str,
        device: str = "cpu",
        preload: bool = True,
        mean_pooling: bool = False,
        disable: bool = False,
        nfeats=None,
        no_model=False,
    ):
        assert not mean_pooling
        self.mean_pooling = mean_pooling
        self.modelname = modelname
        self.embeddings_folder = os.path.join(
            "datasets/annotations", dataname, "text_embeddings"
        )
        self.cache = {}
        self.device = device
        self.disable = disable
        self.dataname = dataname
        self.no_model = no_model

        if preload and not disable:
            self.load_embeddings()
        else:
            self.embeddings_index = {}

    def __contains__(self, text):
        return text in self.embeddings_index

    def get_model(self):
        model = getattr(self, "model", None)
        if model is None:
            model = self.load_model()
        return model

    def __call__(self, texts):
        if self.disable:
            return texts

        squeeze = False
        if isinstance(texts, str):
            texts = [texts]
            squeeze = True

        x_dict_lst = []
        # one at a time here
        for text in texts:
            # Precomputed in advance
            if text in self:
                x_dict = self.get_embedding(text)
            # Already computed during the session
            elif text in self.cache:
                x_dict = self.cache[text]
            # Load the text model (if not already loaded) + compute on the fly
            else:
                if self.no_model:
                    raise ValueError("This text is not precomputed")
                model = self.get_model()
                x_dict = model(text)
                self.cache[text] = x_dict
            x_dict_lst.append(x_dict)

        if squeeze:
            return x_dict_lst[0]
        return x_dict_lst

    def load_model(self):
        self.model = TextToEmb(
            self.modelname, mean_pooling=self.mean_pooling, device=self.device
        )
        return self.model

    def load_embeddings(self):
        # loading can work even with sentence embeddings
        self.embeddings_big = torch.from_numpy(
            np.load(os.path.join(self.embeddings_folder, self.modelname + ".npy"))
        ).to(dtype=torch.float, device=self.device)
        self.embeddings_slice = np.load(
            os.path.join(self.embeddings_folder, self.modelname + "_slice.npy")
        )
        self.embeddings_index = load_json(
            os.path.join(self.embeddings_folder, self.modelname + "_index.json")
        )
        self.text_dim = self.embeddings_big.shape[-1]

    def get_embedding(self, text):
        # Precomputed in advance
        index = self.embeddings_index[text]
        begin, end = self.embeddings_slice[index]
        embedding = self.embeddings_big[begin:end]
        x_dict = {"x": embedding, "length": len(embedding)}
        return x_dict


def load_json(json_path):
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(path, name)
    return load_json(json_path)


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))


def save_text_embeddings(path, modelname, device="cuda"):
    model = TextToEmb(modelname, device=device)
    annotations = load_annotations(path)

    path = os.path.join(path, TextEmbeddings.name)
    ptpath = os.path.join(path, f"{modelname}.npy")
    slicepath = os.path.join(path, f"{modelname}_slice.npy")
    jsonpath = os.path.join(path, f"{modelname}_index.json")

    # modelname can have folders
    path = os.path.split(ptpath)[0]
    os.makedirs(path, exist_ok=True)

    # fetch all the texts
    all_texts = [""]
    for dico in annotations.values():
        for lst in dico["annotations"]:
            all_texts.append(lst["text"])

    # remove duplicates
    all_texts = list(set(all_texts))

    # batch of N/10
    nb_tokens = []
    all_texts_batched = np.array_split(all_texts, 100)

    nb_tokens_so_far = 0
    big_tensor = []
    index = []
    for all_texts_batch in tqdm(all_texts_batched):
        x_dict = model(list(all_texts_batch))

        if isinstance(x_dict, torch.Tensor):
            # sentence embeddings
            assert len(x_dict.shape) == 2
            tensor = x_dict[:, None]
            nb_tokens = torch.tensor([1 for x in range(len(tensor))])
        else:
            tensor = x_dict["x"]
            nb_tokens = x_dict["length"]

        # remove padding
        tensor_no_padding = [x[:n].cpu() for x, n in zip(tensor, nb_tokens)]
        tensor_concat = torch.cat(tensor_no_padding)

        big_tensor.append(tensor_concat)
        # save where it is
        ends = torch.cumsum(nb_tokens, 0)
        begins = torch.cat((0 * ends[[0]], ends[:-1]))

        # offset
        ends += nb_tokens_so_far
        begins += nb_tokens_so_far
        nb_tokens_so_far += len(tensor_concat)

        index.append(torch.stack((begins, ends)).T)

    big_tensor = torch.cat(big_tensor).cpu().numpy()
    index = torch.cat(index).cpu().numpy()

    np.save(ptpath, big_tensor)
    np.save(slicepath, index)
    print(f"{ptpath} written")
    print(f"{slicepath} written")

    # correspondance
    dico = {txt: i for i, txt in enumerate(all_texts)}
    write_json(dico, jsonpath)
    print(f"{jsonpath} written")
