#!/usr/bin/env python3
"""
Vector extraction using XL-LEXEME model for semantic shift analysis.
Extracts normalized vectors from .csv and saves as pickle files.
"""

import ast
import logging
from typing import List, Tuple, Union, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from WordTransformer import WordTransformer
from InputExample import InputExample
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration constants for the semantic similarity calculator."""

    DEFAULT_MAX_SEQ_LEN: int = 128
    DEFAULT_MODEL_NAME: str = "pierluigic/xl-lexeme"
    DEFAULT_BATCH_SIZE: int = 32


class SemanticSimilarityError(Exception):
    """Base exception for semantic similarity errors."""


def _validate_inputs(
    sentence: str,
    positions: Tuple[int, int],
    max_seq_len: int = Config.DEFAULT_MAX_SEQ_LEN,
) -> None:
    """Validate input parameters."""
    if not sentence or not isinstance(sentence, str):
        raise ValueError("Sentence must be a non-empty string")

    if not isinstance(positions, (tuple, list)) or len(positions) != 2:
        raise ValueError("Positions must be a tuple/list of length 2")

    start, end = positions
    if not (0 <= start <= end <= len(sentence)):
        raise ValueError(
            f"Invalid positions: {positions} for sentence length {len(sentence)}"
        )

    if max_seq_len <= 0:
        raise ValueError(f"Invalid max_seq_len: {max_seq_len}")


def _parse_position(pos: Union[str, Tuple[int, int], Any]) -> Tuple[int, int]:
    """Safely parse position from string or tuple."""
    if isinstance(pos, str):
        try:
            return ast.literal_eval(pos)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid position format: {pos}") from e
    elif isinstance(pos, (tuple, list)) and len(pos) == 2:
        return tuple(pos)
    else:
        raise ValueError(
            f"Position must be string or tuple/list of length 2, got {type(pos)}"
        )


class SemanticSimilarityCalculator:
    """Calculator for semantic similarity and vector extraction using XL-DURel model."""

    def __init__(
        self, model_name: str = Config.DEFAULT_MODEL_NAME, device: str = "auto"
    ):
        """
        Initialize calculator with XL-DURel model.

        Args:
            model_name: HuggingFace model name (default: "sachinn1/xl-durel")
            device: Device to use ('auto', 'cpu', 'cuda', 'cuda:0', etc.) (default: 'auto')
        """
        try:
            logger.info(f"Loading model: {model_name}")

            # Determine device
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            logger.info(f"Using device: {self.device}")

            # Load model with specified device
            self.model = WordTransformer(model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            raise SemanticSimilarityError(
                f"Failed to load model {model_name}: {e}"
            ) from e

    def compute_similarity(
        self,
        sentence1: str,
        sentence2: str,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int],
    ) -> float:
        """
        Compute semantic similarity between two sentences (convenience method for single comparisons).

        This method creates a batch of size 1 and delegates to compute_batch_similarities.

        Args:
            sentence1: First sentence string
            sentence2: Second sentence string
            pos1: Tuple (start, end) of target word in sentence1
            pos2: Tuple (start, end) of target word in sentence2

        Returns:
            Cosine similarity score between -1 and 1
        """
        # Create single-row DataFrame
        df = pd.DataFrame(
            {
                "sentence1": [sentence1],
                "sentence2": [sentence2],
                "position1": [pos1],
                "position2": [pos2],
            }
        )

        # Use batch processing with batch size 1
        result_df = self.compute_batch_similarities(df, batch_size=1)

        # Return single similarity value
        return result_df["similarity"].iloc[0]

    def compute_batch_similarities(
        self, df: pd.DataFrame, batch_size: int = Config.DEFAULT_BATCH_SIZE
    ) -> pd.DataFrame:
        """
        Compute similarities for a batch of sentence pairs.

        Args:
            df: DataFrame with columns: sentence1, sentence2, position1, position2
            batch_size: Batch size for processing (default: 32)

        Returns:
            DataFrame with added 'similarity' column
        """
        # Pre-process all sentences first
        examples1 = []
        examples2 = []

        for idx, row in df.iterrows():
            try:
                # Ensure we have proper string values
                sent1 = str(row["sentence1"])
                sent2 = str(row["sentence2"])
                pos1 = _parse_position(row["position1"])
                pos2 = _parse_position(row["position2"])

                example1 = InputExample(texts=sent1, positions=[pos1[0], pos1[1]])
                example2 = InputExample(texts=sent2, positions=[pos2[0], pos2[1]])

                examples1.append(example1)
                examples2.append(example2)
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                examples1.append(None)
                examples2.append(None)

        # Batch encode all examples
        try:
            embeddings1 = self.model.encode(
                examples1,
                batch_size=batch_size,
            )
            embeddings2 = self.model.encode(
                examples2,
                batch_size=batch_size,
            )

            # Compute similarities in batch
            similarities = cosine_similarity(embeddings1, embeddings2).diagonal()
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            similarities = [np.nan] * len(df)

        result_df = df.copy()
        result_df["similarity"] = similarities
        return result_df

    def extract_vectors_for_year(
        self, df: pd.DataFrame, year: int, batch_size: int = 2048
    ) -> dict:
        """
        Extract normalized vectors for a specific year, grouped by word.

        Args:
            df: DataFrame with columns: word, year, sentence, pos_start, pos_end
            year: Year to process (1997 or 2018)
            batch_size: Batch size for processing (default: 128)

        Returns:
            Dictionary mapping word to numpy array of normalized vectors
        """
        year_df = df[df["year"] == year]
        word_vectors = {}

        for word, word_df in tqdm(
            year_df.groupby("word"), desc=f"Processing {year} words"
        ):
            # Process all sentences for this word in one batch
            examples = []
            for _, row in word_df.iterrows():
                example = InputExample(
                    texts=row["sentence"], positions=[row["pos_start"], row["pos_end"]]
                )
                examples.append(example)

            # Batch encode with larger batch size
            vectors = self.model.encode(
                examples,
                batch_size=batch_size,
            )

            # Normalize vectors to unit length
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / norms

            word_vectors[word] = vectors

        return word_vectors

    def get_device_info(self) -> dict:
        """Get information about the current device."""
        info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info.update(
                {
                    "cuda_device_count": torch.cuda.device_count(),
                    "cuda_current_device": torch.cuda.current_device(),
                    "cuda_device_name": torch.cuda.get_device_name(0),
                    "cuda_memory_total": torch.cuda.get_device_properties(
                        0
                    ).total_memory,
                    "cuda_memory_allocated": torch.cuda.memory_allocated(0),
                    "cuda_memory_reserved": torch.cuda.memory_reserved(0),
                }
            )

        return info


def main():
    """Extract vectors from word_matches_spacy.csv and save as pickle files."""

    # Load data
    print("Loading word_matches_spacy.csv...")
    df = pd.read_csv("word_matches_spacy.csv")
    print(f"Loaded {len(df)} sentence matches")

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        device_choice = "auto"  # Will use GPU
    else:
        print("CUDA not available, using CPU")
        device_choice = "cpu"

    calculator = SemanticSimilarityCalculator(device=device_choice)
    print(f"Calculator initialized on device: {calculator.device}")
    print()

    # Process 1997 data
    print("Processing 1997 sentences...")
    T1_vectors = calculator.extract_vectors_for_year(df, 1997, batch_size=2048)

    # Process 2018 data
    print("\nProcessing 2018 sentences...")
    T2_vectors = calculator.extract_vectors_for_year(df, 2018, batch_size=2048)

    # Save pickle files
    import pickle

    print("\nSaving vector files...")
    with open("T1_vecs.pkl", "wb") as f:
        pickle.dump(T1_vectors, f)
    with open("T2_vecs.pkl", "wb") as f:
        pickle.dump(T2_vectors, f)

    # Print statistics
    print(
        f"\nT1_vecs.pkl: {len(T1_vectors)} words, {sum(len(v) for v in T1_vectors.values())} vectors"
    )
    print(
        f"T2_vecs.pkl: {len(T2_vectors)} words, {sum(len(v) for v in T2_vectors.values())} vectors"
    )

    # Verify normalization
    if T1_vectors:
        first_word = list(T1_vectors.keys())[0]
        first_vector = T1_vectors[first_word][0]
        norm = np.linalg.norm(first_vector)
        print(f"\nVerification: First vector norm = {norm:.6f} (should be ~1.0)")

    print("\nVector extraction complete!")


if __name__ == "__main__":
    main()
