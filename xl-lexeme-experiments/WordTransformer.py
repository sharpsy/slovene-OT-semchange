import json
import logging
import os
from collections import OrderedDict
import numpy as np
from numpy import ndarray
import torch
from torch import nn, Tensor, device
from tqdm.autonotebook import trange
from pathlib import Path
from typing import Any, Iterable, Union, List, Dict, Tuple
from sentence_transformers import __MODEL_HUB_ORGANIZATION__
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.model_card import (
    SentenceTransformerModelCardData,
)
from sentence_transformers.util import (
    import_from_string,
    batch_to_device,
)
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import __version__
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.models import Normalize
from sentence_transformers.util import (
    get_device_name,
    is_sentence_transformer_model,
    load_dir_path,
    load_file_path,
)

logger = logging.getLogger(__name__)


class WordTransformer(nn.Sequential):
    """
    Loads or create a SentenceTransformer model, that can be used to map sentences / text to embeddings.

    :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from Huggingface models repository with that name.
    :param modules: This parameter can be used to create custom SentenceTransformer models from scratch.
    :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if a GPU can be used.
    :param cache_folder: Path to store models
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        modules: Iterable[nn.Module] | None = None,
        device: str | None = None,
        prompts: dict[str, str] | None = None,
        default_prompt_name: str | None = None,
        similarity_fn_name: str | SimilarityFunction | None = None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        use_auth_token: bool | str | None = None,
        truncate_dim: int | None = None,
        model_kwargs: dict[str, any] | None = None,
        tokenizer_kwargs: dict[str, any] | None = None,
        config_kwargs: dict[str, any] | None = None,
        model_card_data: SentenceTransformerModelCardData | None = None,
    ) -> None:
        # Note: self._load_sbert_model can also update `self.prompts` and `self.default_prompt_name`
        self.prompts = prompts or {}
        self.default_prompt_name = default_prompt_name
        self.similarity_fn_name = similarity_fn_name
        self.trust_remote_code = trust_remote_code
        self.truncate_dim = truncate_dim
        self.model_card_data = model_card_data or SentenceTransformerModelCardData()
        self.module_kwargs = None
        self._model_card_vars = {}
        self._model_card_text = None
        self._model_config = {}

        if cache_folder is None:
            cache_folder = os.getenv("SENTENCE_TRANSFORMERS_HOME")

        if device is None:
            device = get_device_name()
            logger.info(f"Use pytorch device_name: {device}")

        if model_name_or_path is not None and model_name_or_path != "":
            logger.info(f"Load pretrained SentenceTransformer: {model_name_or_path}")

            # Old models that don't belong to any organization
            basic_transformer_models = [
                "albert-base-v1",
                "albert-base-v2",
                "albert-large-v1",
                "albert-large-v2",
                "albert-xlarge-v1",
                "albert-xlarge-v2",
                "albert-xxlarge-v1",
                "albert-xxlarge-v2",
                "bert-base-cased-finetuned-mrpc",
                "bert-base-cased",
                "bert-base-chinese",
                "bert-base-german-cased",
                "bert-base-german-dbmdz-cased",
                "bert-base-german-dbmdz-uncased",
                "bert-base-multilingual-cased",
                "bert-base-multilingual-uncased",
                "bert-base-uncased",
                "bert-large-cased-whole-word-masking-finetuned-squad",
                "bert-large-cased-whole-word-masking",
                "bert-large-cased",
                "bert-large-uncased-whole-word-masking-finetuned-squad",
                "bert-large-uncased-whole-word-masking",
                "bert-large-uncased",
                "camembert-base",
                "ctrl",
                "distilbert-base-cased-distilled-squad",
                "distilbert-base-cased",
                "distilbert-base-german-cased",
                "distilbert-base-multilingual-cased",
                "distilbert-base-uncased-distilled-squad",
                "distilbert-base-uncased-finetuned-sst-2-english",
                "distilbert-base-uncased",
                "distilgpt2",
                "distilroberta-base",
                "gpt2-large",
                "gpt2-medium",
                "gpt2-xl",
                "gpt2",
                "openai-gpt",
                "roberta-base-openai-detector",
                "roberta-base",
                "roberta-large-mnli",
                "roberta-large-openai-detector",
                "roberta-large",
                "t5-11b",
                "t5-3b",
                "t5-base",
                "t5-large",
                "t5-small",
                "transfo-xl-wt103",
                "xlm-clm-ende-1024",
                "xlm-clm-enfr-1024",
                "xlm-mlm-100-1280",
                "xlm-mlm-17-1280",
                "xlm-mlm-en-2048",
                "xlm-mlm-ende-1024",
                "xlm-mlm-enfr-1024",
                "xlm-mlm-enro-1024",
                "xlm-mlm-tlm-xnli15-1024",
                "xlm-mlm-xnli15-1024",
                "xlm-roberta-base",
                "xlm-roberta-large-finetuned-conll02-dutch",
                "xlm-roberta-large-finetuned-conll02-spanish",
                "xlm-roberta-large-finetuned-conll03-english",
                "xlm-roberta-large-finetuned-conll03-german",
                "xlm-roberta-large",
                "xlnet-base-cased",
                "xlnet-large-cased",
            ]

            if not os.path.exists(model_name_or_path):
                # Not a path, load from hub
                if "\\" in model_name_or_path or model_name_or_path.count("/") > 1:
                    raise ValueError(f"Path {model_name_or_path} not found")

                if (
                    "/" not in model_name_or_path
                    and model_name_or_path.lower() not in basic_transformer_models
                ):
                    # A model from sentence-transformers
                    model_name_or_path = (
                        __MODEL_HUB_ORGANIZATION__ + "/" + model_name_or_path
                    )

            if is_sentence_transformer_model(
                model_name_or_path,
                token,
                cache_folder=cache_folder,
                revision=revision,
                local_files_only=local_files_only,
            ):
                modules, self.module_kwargs = self._load_sbert_model(
                    model_name_or_path,
                    token=token,
                    cache_folder=cache_folder,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs,
                    config_kwargs=config_kwargs,
                )
            else:
                modules = self._load_auto_model(
                    model_name_or_path,
                    token=token,
                    cache_folder=cache_folder,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs,
                    config_kwargs=config_kwargs,
                )

        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict(
                [(str(idx), module) for idx, module in enumerate(modules)]
            )

        super().__init__(modules)

        # Ensure all tensors in the model are of the same dtype as the first tensor
        # This is necessary if the first module has been given a lower precision via
        # model_kwargs["torch_dtype"]. The rest of the model should be loaded in the same dtype
        # See #2887 for more details
        try:
            dtype = next(self.parameters()).dtype
            self.to(dtype)
        except StopIteration:
            pass

        self.to(device)
        self.is_hpu_graph_enabled = False

        if (
            self.default_prompt_name is not None
            and self.default_prompt_name not in self.prompts
        ):
            raise ValueError(
                f"Default prompt name '{self.default_prompt_name}' not found in the configured prompts "
                f"dictionary with keys {list(self.prompts.keys())!r}."
            )

        if self.prompts:
            logger.info(
                f"{len(self.prompts)} prompts are loaded, with the keys: {list(self.prompts.keys())}"
            )
        if self.default_prompt_name:
            logger.warning(
                f"Default prompt name is set to '{self.default_prompt_name}'. "
                "This prompt will be applied to all `encode()` calls, except if `encode()` "
                "is called with `prompt` or `prompt_name` parameters."
            )

        # Ideally, INSTRUCTOR models should set `include_prompt=False` in their pooling configuration, but
        # that would be a breaking change for users currently using the InstructorEmbedding project.
        # So, instead we hardcode setting it for the main INSTRUCTOR models, and otherwise give a warning if we
        # suspect the user is using an INSTRUCTOR model.
        if model_name_or_path in (
            "hkunlp/instructor-base",
            "hkunlp/instructor-large",
            "hkunlp/instructor-xl",
        ):
            self.set_pooling_include_prompt(include_prompt=False)
        elif (
            model_name_or_path
            and "/" in model_name_or_path
            and "instructor" in model_name_or_path.split("/")[1].lower()
        ):
            if any(
                [
                    module.include_prompt
                    for module in self
                    if isinstance(module, Pooling)
                ]
            ):
                logger.warning(
                    "Instructor models require `include_prompt=False` in the pooling configuration. "
                    "Either update the model configuration or call `model.set_pooling_include_prompt(False)` after loading the model."
                )

        # Pass the model to the model card data for later use in generating a model card upon saving this model
        self.model_card_data.register_model(self)

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self.device

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(
            0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.forward(features)

                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(
                        out_features[output_value], out_features["attention_mask"]
                    ):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0 : last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features["sentence_embedding"])):
                        row = {
                            name: out_features[name][sent_idx] for name in out_features
                        }
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(
                            embeddings, p=2, dim=1
                        )

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def get_max_seq_length(self):
        """
        Returns the maximal sequence length for input the model accepts. Longer inputs will be truncated
        """
        if hasattr(self._first_module(), "max_seq_length"):
            return self._first_module().max_seq_length

        return None

    def center_sentence(self, input_ids, positions, max_seq_len):
        left = input_ids[: positions[0]]
        right = input_ids[positions[1] :]

        overflow_left = len(left) - int(
            (max_seq_len - len(input_ids[positions[0] : positions[1]])) / 2
        )
        overflow_right = len(right) - int(
            (max_seq_len - len(input_ids[positions[0] : positions[1]])) / 2
        )

        if overflow_left > 0 and overflow_right > 0:
            left = left[overflow_left:]
            right = right[: len(right) - overflow_right]

        elif overflow_left > 0 and overflow_right <= 0:
            left = left[overflow_left:]

        else:
            right = right[: len(right) - overflow_right]

        return left + input_ids[positions[0] : positions[1]] + right

    def tokenize_sentence(self, sentence, positions):
        left, target, right = (
            sentence[: positions[0]],
            sentence[positions[0] : positions[1]],
            sentence[positions[1] :],
        )

        token_positions = [0, 0]
        tokens = []

        if left:
            tokens += self._first_module().tokenizer.tokenize(left)
        token_positions[0] = len(tokens)
        tokens += self._first_module().tokenizer.tokenize("<t>")
        target_subtokens = self._first_module().tokenizer.tokenize(target)
        tokens += target_subtokens
        tokens += self._first_module().tokenizer.tokenize("</t>")
        token_positions[1] = len(tokens)
        if right:
            tokens += self._first_module().tokenizer.tokenize(right)

        return tokens, token_positions

    def tokenize_batch(self, batch, sentid=None):
        max_example_length = 0

        tokenized = {"input_ids": [], "attention_mask": []}

        for example in batch:
            n_extra_tokens = 2

            if sentid is not None:
                tokens, token_positions = self.tokenize_sentence(
                    example.texts[sentid], example.positions[sentid]
                )
            else:
                tokens, token_positions = self.tokenize_sentence(
                    example.texts, example.positions
                )

            len_input = len(tokens) + n_extra_tokens

            if len_input > self._first_module().max_seq_length:
                tokens = self.center_sentence(
                    tokens,
                    token_positions,
                    self._first_module().max_seq_length - n_extra_tokens,
                )

            tokens = (
                [self._first_module().tokenizer.cls_token]
                + tokens
                + [self._first_module().tokenizer.sep_token]
            )

            input_ids = self._first_module().tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)

            tokenized["input_ids"].append(input_ids)
            tokenized["attention_mask"].append(attention_mask)

            if len(input_ids) > max_example_length:
                max_example_length = len(input_ids)

        for j in range(len(tokenized["input_ids"])):
            tokenized["input_ids"][j] = tokenized["input_ids"][j] + [
                self._first_module().tokenizer.convert_tokens_to_ids(
                    [self._first_module().tokenizer.pad_token]
                )[0]
            ] * (max_example_length - len(tokenized["input_ids"][j]))
            tokenized["attention_mask"][j] = tokenized["attention_mask"][j] + [0] * (
                max_example_length - len(tokenized["attention_mask"][j])
            )

        tokenized["input_ids"] = torch.tensor(tokenized["input_ids"]).to(self.device)
        tokenized["attention_mask"] = torch.tensor(tokenized["attention_mask"]).to(
            self.device
        )

        return tokenized

    def tokenize(
        self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]], sentid=None
    ):
        """
        Tokenizes the texts
        """
        return self.tokenize_batch(texts, sentid)

    def get_sentence_features(self, *features):
        return self._first_module().get_sentence_features(*features)

    def get_sentence_embedding_dimension(self):
        for mod in reversed(self._modules.values()):
            sent_embedding_dim_method = getattr(
                mod, "get_sentence_embedding_dimension", None
            )
            if callable(sent_embedding_dim_method):
                return sent_embedding_dim_method()
        return None

    def _first_module(self):
        """Returns the first module of this sequential embedder"""
        return self._modules[next(iter(self._modules))]

    def _last_module(self):
        """Returns the last module of this sequential embedder"""
        return self._modules[next(reversed(self._modules))]

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)
        labels = [example.label for example in batch]

        labels = torch.tensor(labels).to(self.device)

        sentence_features = []

        for idx in range(num_texts):
            tokenized = self.tokenize(batch, idx)
            batch_to_device(tokenized, self.device)
            sentence_features.append(tokenized)

        return sentence_features, labels

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def evaluate(self, evaluator: SentenceEvaluator, output_path: str = None):
        """
        Evaluate the model

        :param evaluator:
            the evaluator
        :param output_path:
            the evaluator can write the results to this path
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        return evaluator(self, output_path)

    def _load_auto_model(self, model_name_or_path):
        """
        Creates a simple Transformer + Mean Pooling model and returns the modules
        """
        logging.warning(
            "No sentence-transformers model found with name {}. Creating a new one with MEAN pooling.".format(
                model_name_or_path
            )
        )
        transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(
            transformer_model.get_word_embedding_dimension(), "mean"
        )
        return [transformer_model, pooling_model]

    def _load_sbert_model(
        self,
        model_name_or_path: str,
        token: bool | str | None,
        cache_folder: str | None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, nn.Module]:
        """
        Loads a full SentenceTransformer model using the modules.json file.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model.
            token (Optional[Union[bool, str]]): The token to use for the model.
            cache_folder (Optional[str]): The folder to cache the model.
            revision (Optional[str], optional): The revision of the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            local_files_only (bool, optional): Whether to use only local files. Defaults to False.
            model_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the model. Defaults to None.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the tokenizer. Defaults to None.
            config_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the config. Defaults to None.

        Returns:
            OrderedDict[str, nn.Module]: An ordered dictionary containing the modules of the model.
        """
        # Check if the config_sentence_transformers.json file exists (exists since v2 of the framework)
        config_sentence_transformers_json_path = load_file_path(
            model_name_or_path,
            "config_sentence_transformers.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        if config_sentence_transformers_json_path is not None:
            with open(config_sentence_transformers_json_path) as fIn:
                self._model_config = json.load(fIn)

            if (
                "__version__" in self._model_config
                and "sentence_transformers" in self._model_config["__version__"]
                and self._model_config["__version__"]["sentence_transformers"]
                > __version__
            ):
                logger.warning(
                    "You try to use a model that was created with version {}, however, your version is {}. This might cause unexpected behavior or errors. In that case, try to update to the latest version.\n\n\n".format(
                        self._model_config["__version__"]["sentence_transformers"],
                        __version__,
                    )
                )

            # Set score functions & prompts if not already overridden by the __init__ calls
            if self.similarity_fn_name is None:
                self.similarity_fn_name = self._model_config.get(
                    "similarity_fn_name", None
                )
            if not self.prompts:
                self.prompts = self._model_config.get("prompts", {})
            if not self.default_prompt_name:
                self.default_prompt_name = self._model_config.get(
                    "default_prompt_name", None
                )

        # Check if a readme exists
        model_card_path = load_file_path(
            model_name_or_path,
            "README.md",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        if model_card_path is not None:
            try:
                with open(model_card_path, encoding="utf8") as fIn:
                    self._model_card_text = fIn.read()
            except Exception:
                pass

        # Load the modules of sentence transformer
        modules_json_path = load_file_path(
            model_name_or_path,
            "modules.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        with open(modules_json_path) as fIn:
            modules_config = json.load(fIn)

        modules = OrderedDict()
        module_kwargs = OrderedDict()
        for module_config in modules_config:
            class_ref = module_config["type"]
            module_class = self._load_module_class_from_ref(
                class_ref, model_name_or_path, trust_remote_code, revision, model_kwargs
            )

            # For Transformer, don't load the full directory, rely on `transformers` instead
            # But, do load the config file first.
            if module_config["path"] == "":
                kwargs = {}
                for config_name in [
                    "sentence_bert_config.json",
                    "sentence_roberta_config.json",
                    "sentence_distilbert_config.json",
                    "sentence_camembert_config.json",
                    "sentence_albert_config.json",
                    "sentence_xlm-roberta_config.json",
                    "sentence_xlnet_config.json",
                ]:
                    config_path = load_file_path(
                        model_name_or_path,
                        config_name,
                        token=token,
                        cache_folder=cache_folder,
                        revision=revision,
                        local_files_only=local_files_only,
                    )
                    if config_path is not None:
                        with open(config_path) as fIn:
                            kwargs = json.load(fIn)
                            # Don't allow configs to set trust_remote_code
                            if (
                                "model_args" in kwargs
                                and "trust_remote_code" in kwargs["model_args"]
                            ):
                                kwargs["model_args"].pop("trust_remote_code")
                            if (
                                "tokenizer_args" in kwargs
                                and "trust_remote_code" in kwargs["tokenizer_args"]
                            ):
                                kwargs["tokenizer_args"].pop("trust_remote_code")
                            if (
                                "config_args" in kwargs
                                and "trust_remote_code" in kwargs["config_args"]
                            ):
                                kwargs["config_args"].pop("trust_remote_code")
                        break

                hub_kwargs = {
                    "token": token,
                    "trust_remote_code": trust_remote_code,
                    "revision": revision,
                    "local_files_only": local_files_only,
                }
                # 3rd priority: config file
                if "model_args" not in kwargs:
                    kwargs["model_args"] = {}
                if "tokenizer_args" not in kwargs:
                    kwargs["tokenizer_args"] = {}
                if "config_args" not in kwargs:
                    kwargs["config_args"] = {}

                # 2nd priority: hub_kwargs
                kwargs["model_args"].update(hub_kwargs)
                kwargs["tokenizer_args"].update(hub_kwargs)
                kwargs["config_args"].update(hub_kwargs)

                # 1st priority: kwargs passed to SentenceTransformer
                if model_kwargs:
                    kwargs["model_args"].update(model_kwargs)
                if tokenizer_kwargs:
                    kwargs["tokenizer_args"].update(tokenizer_kwargs)
                if config_kwargs:
                    kwargs["config_args"].update(config_kwargs)

                # Try to initialize the module with a lot of kwargs, but only if the module supports them
                # Otherwise we fall back to the load method
                # try:
                module = module_class(
                    model_name_or_path, cache_dir=cache_folder, **kwargs
                )
                # except TypeError:
                #     module = module_class.load(model_name_or_path)
            else:
                # Normalize does not require any files to be loaded
                if module_class == Normalize:
                    module_path = None
                else:
                    module_path = load_dir_path(
                        model_name_or_path,
                        module_config["path"],
                        token=token,
                        cache_folder=cache_folder,
                        revision=revision,
                        local_files_only=local_files_only,
                    )
                module = module_class.load(module_path)

            modules[module_config["name"]] = module
            module_kwargs[module_config["name"]] = module_config.get("kwargs", [])

        if revision is None:
            path_parts = Path(modules_json_path)
            if len(path_parts.parts) >= 2:
                revision_path_part = Path(modules_json_path).parts[-2]
                if len(revision_path_part) == 40:
                    revision = revision_path_part
        self.model_card_data.set_base_model(model_name_or_path, revision=revision)
        return modules, module_kwargs

    def _load_module_class_from_ref(
        self,
        class_ref: str,
        model_name_or_path: str,
        trust_remote_code: bool,
        revision: str | None,
        model_kwargs: dict[str, Any] | None,
    ) -> nn.Module:
        # If the class is from sentence_transformers, we can directly import it,
        # otherwise, we try to import it dynamically, and if that fails, we fall back to the default import
        if class_ref.startswith("sentence_transformers."):
            return import_from_string(class_ref)

        return import_from_string(class_ref)

    @property
    def device(self) -> device:
        """
        Get torch.device from module, assuming that the whole module has one device.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
                tuples = [
                    (k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)
                ]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].device

    @property
    def tokenizer(self):
        """
        Property to get the tokenizer that is used by this model
        """
        return self._first_module().tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        """
        Property to set the tokenizer that is should used by this model
        """
        self._first_module().tokenizer = value

    @property
    def max_seq_length(self):
        """
        Property to get the maximal input sequence length for the model. Longer inputs will be truncated.
        """
        return self._first_module().max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value):
        """
        Property to set the maximal input sequence length for the model. Longer inputs will be truncated.
        """
        self._first_module().max_seq_length = value
