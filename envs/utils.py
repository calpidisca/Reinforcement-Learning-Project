"""Utility functions for superpermutation environments."""

import numpy as np
from typing import List, Tuple
from itertools import permutations


def get_all_permutations(n: int) -> List[Tuple[int, ...]]:
    """
    Generate all permutations of the alphabet {1, 2, ..., n}.
    
    Args:
        n: Size of the alphabet
        
    Returns:
        List of all permutations as tuples
    """
    alphabet = list(range(1, n + 1))
    return list(permutations(alphabet))


def max_overlap(seq: List[int], word: List[int]) -> int:
    """
    Compute the maximum overlap length between seq and word.
    
    The overlap is the length ℓ such that the last ℓ symbols of seq
    equal the first ℓ symbols of word.
    
    Args:
        seq: Current sequence
        word: Word to check overlap with
        
    Returns:
        Maximum overlap length (0 to min(len(seq), len(word)))
    """
    if len(seq) == 0 or len(word) == 0:
        return 0
    
    max_len = min(len(seq), len(word))
    for overlap_len in range(max_len, 0, -1):
        if seq[-overlap_len:] == word[:overlap_len]:
            return overlap_len
    return 0


def merge_with_overlap(seq: List[int], word: List[int]) -> List[int]:
    """
    Merge word into seq using maximum overlap.
    
    Args:
        seq: Current sequence
        word: Word to merge
        
    Returns:
        Merged sequence: seq + word[overlap:]
    """
    overlap = max_overlap(seq, word)
    return seq + word[overlap:]


def update_coverage_for_string(
    string: List[int],
    n: int,
    perms: List[Tuple[int, ...]],
    coverage: np.ndarray,
    search_window: int | None = None,
) -> int:
    """
    Update coverage array by scanning string for contiguous substrings of length n.
    
    Args:
        string: The sequence to scan
        n: Length of permutations
        perms: List of all permutations (tuples)
        coverage: Boolean array of shape (len(perms),) to update
        search_window: If not None, only scan the last search_window symbols
        
    Returns:
        Number of newly discovered permutations (positions that changed from False to True)
    """
    if len(string) < n:
        return 0
    
    # Determine search range
    if search_window is not None:
        start_idx = max(0, len(string) - search_window)
    else:
        start_idx = 0
    
    delta_new = 0
    # Scan all length-n substrings in the search window
    for i in range(start_idx, len(string) - n + 1):
        substring = tuple(string[i:i+n])
        # Find index in perms
        try:
            idx = perms.index(substring)
            if not coverage[idx]:
                coverage[idx] = True
                delta_new += 1
        except ValueError:
            # Substring is not a valid permutation, skip
            pass
    
    return delta_new


def flatten_symbol_observation(obs: dict, n: int, m: int) -> np.ndarray:
    """
    Flatten symbol environment observation for neural networks.
    
    Args:
        obs: Observation dict with keys "suffix", "coverage", "length"
        n: Alphabet size
        m: Number of permutations
        
    Returns:
        1D numpy array
    """
    suffix = obs["suffix"].astype(np.float32)
    coverage = obs["coverage"].astype(np.float32)
    length = obs["length"].astype(np.float32)
    return np.concatenate([suffix, coverage, length])


def flatten_word_observation(obs: dict, n: int, m: int) -> np.ndarray:
    """
    Flatten word environment observation for neural networks.
    
    Args:
        obs: Observation dict with keys "coverage", "costs", "length"
        n: Alphabet size
        m: Number of permutations
        
    Returns:
        1D numpy array
    """
    coverage = obs["coverage"].astype(np.float32)
    costs = obs["costs"].astype(np.float32)
    length = obs["length"].astype(np.float32)
    return np.concatenate([coverage, costs, length])


def canonicalize_superperm(sequence: List[int], n: int) -> List[int]:
    """
    Canonicalize a superpermutation by relabeling to a standard form.
    
    Approach:
    1. Find the first contiguous block of length n that is a permutation of {1,...,n}
    2. Use that block to define a relabeling mapping
    3. Apply the mapping to the entire sequence
    
    Args:
        sequence: The superpermutation sequence
        n: Alphabet size
        
    Returns:
        Canonicalized sequence
    """
    if len(sequence) < n:
        return sequence.copy()
    
    # Find first permutation block
    mapping = None
    for i in range(len(sequence) - n + 1):
        block = sequence[i:i+n]
        # Check if block is a permutation of {1,...,n}
        if set(block) == set(range(1, n + 1)) and len(block) == len(set(block)):
            # Create mapping: block[j] -> j+1
            mapping = {block[j]: j + 1 for j in range(n)}
            break
    
    # If no permutation block found, return original
    if mapping is None:
        return sequence.copy()
    
    # Apply mapping
    canonical = [mapping.get(symbol, symbol) for symbol in sequence]
    return canonical


