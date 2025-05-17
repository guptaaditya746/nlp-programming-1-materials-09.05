#!/usr/bin/python3
from tqdm import tqdm
import re
import json
import os
from collections import  Counter
from typing import List,  Tuple, Optional


GROUP = "27"  # TODO: write in your group number


def load_imdb_dataset(file_path: str="./imdb.txt", small_dataset=True) -> List[str]:
    """ This function loads the IMDB dataset from the txt file.
    Args:
        file_path (str): The path to the json file.
        p (int): Percentage of the dataset to use.
    Returns:
        list: A list of texts from the dataset.
    """
    # this function was implemented for you
    with open(file_path, 'r') as f:
        dataset = f.readlines()
    dataset = dataset[:100] if small_dataset else dataset
    print(f"Loaded {len(dataset)} documents")
    return dataset


class BPETokenizer:
    def __init__(self):
        """Initialize the BPE tokenizer."""
        self.vocab = {} 
        self.merges = [] 
    ################################# BPE merge rule finding #########################################
    ############################## Lecture slides: NLP:III-43--51 ####################################
    def pre_tokenize(self, text: str) -> List[str]:
        """Preprocess the text: normalize, clean HTML, and tokenize with regex."""
        # Lowercase
        text = text.lower()

        # Remove HTML tags like <br />
        text = re.sub(r'<[^>]+>', ' ', text)

        # Replace multiple punctuation (e.g. !!!, ???) with single
        text = re.sub(r'([!?.,])\1+', r'\1', text)

        # Normalize dashes (like em-dash, en-dash, hyphens)
        text = re.sub(r'[–—−]', '-', text)

        # Remove unwanted characters (like fancy quotes, non-breaking spaces)
        text = re.sub(r'[“”]', '"', text)
        text = re.sub(r"[‘’]", "'", text)
        text = re.sub(r'\xa0', ' ', text)

        # Handle common contractions and abbreviations
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "can not", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'t", " not", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'m", " am", text)

        # Handle special cases from sample text
        text = re.sub(r'\*\*\*spoilers\*\*\*', ' spoilers ', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+/\d+\b', ' ', text)  # Remove scores like 4/10
        text = re.sub(r'\b\w+-\w+\b', lambda m: m.group(0).replace('-', ' '), text)  # Split hyphenated words
        text = re.sub(r'\.{3,}', '...', text)  # Normalize ellipses
        text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces

        # Handle tokenization using regex:
        # Match:
        #   1. Words with apostrophes (contractions like "don't")
        #   2. Words (letters or digits)
        #   3. Punctuation as standalone tokens
        #   4. Common emoticons/symbols
        pattern = r"""
            \b\w+'\w+|\b\w+\b|[^\w\s]|
            [;:]-?[\)\(dDpP/\\]|  # Basic emoticons
            \d+\.\d+               # Numbers with decimals
        """
        tokens = re.findall(pattern, text, re.VERBOSE)

        return tokens


    def preprocess(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocesses raw texts. For each text, it's tokenized into words using
        `self.pre_tokenize`, words are split into characters, and then all 
        characters for that text are flattened into a single list.
        Args:
            texts: List of raw text strings (documents).
        Returns:
            List of lists, where each inner list contains all character-level tokens 
            for a document, concatenated in order.
            e.g., ["Hello world."] -> [['h', 'e', 'l', 'l', 'o', 'w', 'o', 'r', 'l', 'd', '.']]
            (assuming pre_tokenize results in ["hello", "world."])
        """                
        preprocessed_texts = []
        
        for text in texts:
            # Use custom tokenizer to tokenize the text into strings
            pre_tokenized_text = self.pre_tokenize(text)
            
            # For each word in the pre_tokenized text, split it into characters
            # word_char_lists becomes a list of lists of characters, e.g., [['h','e','l','l','o'], ['w','o','r','l','d','.']]
            word_char_lists = [list(word) for word in pre_tokenized_text]
            
            # Flatten the list of character-lists for the current document
            document_chars = [char for word_list in word_char_lists for char in word_list]
            preprocessed_texts.append(document_chars)
            
        return preprocessed_texts


    def _get_stats(self, preprocessed_texts: List[List[str]]) -> Counter:
        """
        Count subword pair frequencies in the preprocessed texts.
        Each inner list in preprocessed_texts is a sequence of characters/subwords for a document.
        Args:
            preprocessed_texts: List of lists of strings (characters/subwords).
        Returns:
            Counter of subword pair frequencies.
        """
        pairs = Counter()
        # Iterate over each sequence of characters/subwords in the preprocessed texts
        # (each sequence corresponds to an original document)
        for sequence in preprocessed_texts:
            # Iterate through the sequence to form adjacent pairs
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i+1])
                pairs[pair] += 1
        return pairs

    def _merge_pair(self, 
                    preprocessed_texts: List[List[str]], 
                    pair: Tuple[str, str]) -> List[List[str]]:
        """
        Merge all occurrences of a pair in the preprocessed texts.
        
        Args:
            preprocessed_texts: List of lists of tokenized strings
            pair: Tuple of strings (substrings) to merge
            
        Returns:
            Updated preprocessed texts with pairs merged
        """
        merged_texts = []
        merged_token = "".join(pair) # e.g., 'th' from ('t', 'h')
        
        for sequence in preprocessed_texts: # sequence is a List[str], e.g. ['t', 'h', 'e', 'q', 'u', ...]
            new_sequence = []
            i = 0
            while i < len(sequence):
                # Check if the current position and the next form the pair to be merged
                if i < len(sequence) - 1 and (sequence[i], sequence[i+1]) == pair:
                    new_sequence.append(merged_token)
                    i += 2 # Skip the next token as it's part of the merged pair
                else:
                    new_sequence.append(sequence[i])
                    i += 1
            merged_texts.append(new_sequence)
            
        return merged_texts

    def train(self, 
              texts: List[str], 
              max_merges: Optional[int] = None, 
              max_vocab_size: Optional[int] = None) -> None:
        """
        Train the BPE tokenizer on the input texts.
        Algorithm was introduced in the lecture slides (NLP:III-51).
        Args:
            texts: List of text strings for training
            max_merges: Maximum number of merge operations to perform (optional)
            max_vocab_size: Maximum vocabulary size to aim for (optional)
        """ 
        # Lecture slides: NLP:III-43--51
        # 1. Create an initial tokenization of a training corpus 
        # + 3. Split each token into symbols; 
        preprocessed_texts = self.preprocess(texts)

        # 3. initialize vocabulary V with individual characters
        self.vocab = {}
        # preprocessed_texts is List[List[str]], where each inner list is a sequence of characters for a document
        initial_symbols = set()
        for char_sequence in preprocessed_texts:
            for char_token in char_sequence: # char_token is a single character string
                initial_symbols.add(char_token)

        # assign IDs to each token in the vocabulary
        for idx, token in enumerate(sorted(list(initial_symbols))): # Ensure sorted for deterministic IDs
            self.vocab[token] = idx

        # initialize merge rules list
        self.merges = []

        num_merges_done = 0
        
        pbar = None
        if max_merges is not None:
            pbar = tqdm(total=max_merges, desc="Training BPE", unit="merge")
        elif max_vocab_size is not None:
            # Estimate total merges needed to reach max_vocab_size for tqdm
            initial_vocab_len = len(self.vocab)
            if max_vocab_size > initial_vocab_len:
                estimated_total_merges = max_vocab_size - initial_vocab_len
                pbar = tqdm(total=estimated_total_merges, desc="Training BPE (target vocab size)", unit="merge")
            # If max_vocab_size is already met or less than initial, 
            # pbar can remain None, and the loop should terminate quickly or not run if already satisfied.
            # If max_vocab_size is already met or less than initial, pbar remains None,
            # and the loop should terminate quickly.

        while True:
            # Stopping condition 1: max_merges reached
            if max_merges is not None and num_merges_done >= max_merges:
                break
            
            # Stopping condition 2: max_vocab_size reached (target met or exceeded)
            if max_vocab_size is not None and len(self.vocab) >= max_vocab_size:
                break

            stats = self._get_stats(preprocessed_texts)
            if not stats: # No more pairs to merge
                break

            best_pair_info = stats.most_common(1)
            if not best_pair_info: # Should be covered by `if not stats:`
                break
            
            best_pair, count = best_pair_info[0]
            # Optional: if count < 2, could stop, but sticking to explicit limits for now.

            preprocessed_texts = self._merge_pair(preprocessed_texts, best_pair)
            self.merges.append(best_pair)
            new_token = "".join(best_pair)

            if new_token not in self.vocab:
                # Only add to vocab if we haven't reached max_vocab_size
                if max_vocab_size is None or len(self.vocab) < max_vocab_size:
                    self.vocab[new_token] = len(self.vocab)
            
            num_merges_done += 1
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"vocab_size": len(self.vocab), "merges": num_merges_done, "last_pair": f"{best_pair}->{new_token}"})
            elif num_merges_done % 100 == 0: # Fallback print if no tqdm and no max_merges
                 print(f"Merge {num_merges_done}: {best_pair} -> {new_token}, Vocab size: {len(self.vocab)}")

        if pbar:
            pbar.close()
        
        print(f"Training finished. Performed {num_merges_done} merges. Final vocab size: {len(self.vocab)}.")


    ######################################## BPE tokenization #############################################
    ################################## Lecture slides: NLP:III-38--42 #####################################
    def _tokenize_string(self, string: str) -> List[str]:
        """
        Tokenize a single string (word) using the trained merge rules.
        Args:
            string: Input string (word)
        Returns:
            List of subword tokens
        """
        # Implemented correctly: splits to chars, iteratively applies highest priority merge found.
        tokens = list(string)

        while True:
            found_merge = False
            # Iterate through learned merges in order of priority
            for merge_rule in self.merges:
                # Find the first occurrence of this merge rule in the current tokens
                i = 0
                while i < len(tokens) - 1:
                    if (tokens[i], tokens[i+1]) == merge_rule:
                        # Merge and restart the check from the highest priority rule
                        tokens = tokens[:i] + ["".join(merge_rule)] + tokens[i+2:]
                        found_merge = True
                        break # Break inner loop (over positions i)
                    i += 1
                if found_merge:
                    break # Break outer loop (over merge_rules) to restart scan
            # If no merge was found in a full pass through merge rules, we are done
            if not found_merge:
                break
        return tokens




    def tokenize(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize new texts using the trained BPE merge rules.
        Args:
            texts: List of input text strings
        Returns:
            List of lists of strings tokeniz ed into substrings
        """
        if not self.merges:
            raise ValueError("Tokenizer has not been trained. Call train() first.")
            
        result = []
        
        for text in tqdm(texts):
            strings = self.pre_tokenize(text)
            # Flatten the list of subword tokens for each sentence
            tokenized_word_lists = [self._tokenize_string(s) for s in strings] # e.g. [['th','e'], ['q','u','i','ck']]
            # Flatten the list of subword-lists for the current document into a single list of subwords
            document_tokens = [token for sublist in tokenized_word_lists for token in sublist] # e.g. ['th','e','q','u','i','ck']
            result.append(document_tokens)

        return result


    def save(self, path: str=f"./output/bpe_group_{GROUP}.json") -> None:
        """
        Save the trained tokenizer to a file
        Args:
            path: Path to save the tokenizer
        """
        # this method was implemented for you
        if not self.merges:
            raise ValueError("Tokenizer has not been trained. Call train() first.")
        
        serialized_merges = [list(pair) for pair in self.merges]
        tokenizer_data = {
            "vocab": self.vocab,
            "merges": serialized_merges,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=4)
        print(f"Tokenizer saved to {path}")
    

    @classmethod
    def load(cls, path: str=f"./output/bpe_group_{GROUP}.json") -> 'BPETokenizer':
        """
        Load a trained tokenizer from a file.
        
        Args:
            path: Path to load the tokenizer from
            
        Returns:
            Loaded BPETokenizer instance
        """
        # this method was implemented for you
        tokenizer = cls()
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        tokenizer.vocab = tokenizer_data["vocab"]
        tokenizer.merges = [tuple(pair) for pair in tokenizer_data["merges"]]
        print(f"Tokenizer loaded from {path}")
        return tokenizer

def main(num_merges: int=1000, 
         max_vocab_size: Optional[int]=None, 
         small_dataset: bool=False) -> None:
    """
    Main function to train and save the BPE tokenizer.
    
    Args:
        num_merges: Number of merge operations to perform
    """
    dataset = load_imdb_dataset(small_dataset=small_dataset)
    tokenizer = BPETokenizer()
    tokenizer.train(dataset, max_merges=num_merges, max_vocab_size=max_vocab_size)
    tokenizer.save()
    tokenized = tokenizer.tokenize(["The foxes are running quickly.", "this is a movie"])
    with open(f"./output/bpe_test_group-{GROUP}.txt", "w") as f:
        for t in tokenized:
            f.write(f"{t}\n")
################################################
# Tests
import os
import pytest
import sys

def sample_texts():
    return ["the quick brown fox jumps over the lazy dog.",
            "bpe works well for subword tokenization."]

def test_that_the_group_name_is_there():
    import re
    assert re.match(r'^[0-9]{1,2}$', GROUP), \
        "Please write your group name in the variable at the top of the file!"

def test_train_tokenizer() -> BPETokenizer:
    tokenizer = BPETokenizer()
    tokenizer.train(sample_texts(), max_merges=15)
    assert len(tokenizer.merges) == 15, "Tokenizer did not train correctly"
    assert len(tokenizer.vocab) > 0, "Tokenizer vocabulary is empty"
    assert len(tokenizer.merges) > 0, "Tokenizer merges are empty"

def test_preprocess():
    """Test the preprocessing method."""
    tokenizer = BPETokenizer()
    processed = tokenizer.preprocess(sample_texts())
    
    expected = [['t', 'h', 'e', 'q', 'u', 'i', 'c', 'k', 'b', 'r', 'o', 'w', 'n', 'f', 'o', 'x', 
                 'j', 'u', 'm', 'p', 's', 'o', 'v', 'e', 'r', 't', 'h', 'e', 'l', 'a', 'z', 'y', 'd', 'o', 'g', '.'], 
                 ['b', 'p', 'e', 'w', 'o', 'r', 'k', 's', 'w', 'e', 'l', 'l', 'f', 'o', 'r', 
                  's', 'u', 'b', 'w', 'o', 'r', 'd', 't', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n', '.']]
    assert processed == expected

def test_get_stats():
    """Test the _get_stats method for calculating pair frequencies."""
    tokenizer = BPETokenizer()
    # tokenized_texts = [["a", "b", "c"], ["d", "e"]], [["b", "c"], ["d", "e"]]
    tokenized_texts = tokenizer.preprocess(sample_texts())
    # tokenized_texts = [[['h', 'e', 'l', 'l', 'o'], ['w', 'o', 'r', 'l', 'd', '!']], [['h', 'o', 'w'], ['a', 'r', 'e'], ['y', 'o', 'u', '?']]]
    
    stats = tokenizer._get_stats(tokenized_texts)
    
    expected_pairs = Counter({('o', 'r'): 3, ('t', 'h'): 2, ('h', 'e'): 2, ('f', 'o'): 2, ('e', 'l'): 2, ('w', 'o'): 2, 
                              ('e', 'q'): 1, ('q', 'u'): 1, ('u', 'i'): 1, ('i', 'c'): 1, ('c', 'k'): 1, ('k', 'b'): 1, 
                              ('b', 'r'): 1, ('r', 'o'): 1, ('o', 'w'): 1, ('w', 'n'): 1, ('n', 'f'): 1, ('o', 'x'): 1, 
                              ('x', 'j'): 1, ('j', 'u'): 1, ('u', 'm'): 1, ('m', 'p'): 1, ('p', 's'): 1, ('s', 'o'): 1, 
                              ('o', 'v'): 1, ('v', 'e'): 1, ('e', 'r'): 1, ('r', 't'): 1, ('l', 'a'): 1, ('a', 'z'): 1, 
                              ('z', 'y'): 1, ('y', 'd'): 1, ('d', 'o'): 1, ('o', 'g'): 1, ('g', '.'): 1, ('b', 'p'): 1, 
                              ('p', 'e'): 1, ('e', 'w'): 1, ('r', 'k'): 1, ('k', 's'): 1, ('s', 'w'): 1, ('w', 'e'): 1, 
                              ('l', 'l'): 1, ('l', 'f'): 1, ('r', 's'): 1, ('s', 'u'): 1, ('u', 'b'): 1, ('b', 'w'): 1, 
                              ('r', 'd'): 1, ('d', 't'): 1, ('t', 'o'): 1, ('o', 'k'): 1, ('k', 'e'): 1, ('e', 'n'): 1, 
                              ('n', 'i'): 1, ('i', 'z'): 1, ('z', 'a'): 1, ('a', 't'): 1, ('t', 'i'): 1, ('i', 'o'): 1, 
                              ('o', 'n'): 1, ('n', '.'): 1})
    assert stats == expected_pairs

def test_merge_pair():
    """Test the _merge_pair method."""
    tokenizer = BPETokenizer()
    tokenized_texts = [['t', 'h', 'e', 'q', 'u', 'i', 'c', 'k', 'b', 'r', 'o', 'w', 'n', 'f', 'o', 'x', 
                 'j', 'u', 'm', 'p', 's', 'o', 'v', 'e', 'r', 't', 'h', 'e', 'l', 'a', 'z', 'y', 'd', 'o', 'g', '.'], 
                 ['b', 'p', 'e', 'w', 'o', 'r', 'k', 's', 'w', 'e', 'l', 'l', 'f', 'o', 'r', 
                  's', 'u', 'b', 'w', 'o', 'r', 'd', 't', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n', '.']]
    pair = ("o", "r")
    
    merged = tokenizer._merge_pair(tokenized_texts, pair)
    #print('MERGED:', merged)
    expected = [['t', 'h', 'e', 'q', 'u', 'i', 'c', 'k', 'b', 'r', 'o', 'w', 'n', 'f', 'o', 'x', 
                 'j', 'u', 'm', 'p', 's', 'o', 'v', 'e', 'r', 't', 'h', 'e', 'l', 'a', 'z', 'y', 'd', 'o', 'g', '.'], 
                 ['b', 'p', 'e', 'w', 'or', 'k', 's', 'w', 'e', 'l', 'l', 'f', 'or', 
                  's', 'u', 'b', 'w', 'or', 'd', 't', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n', '.']]
    assert merged == expected

def test_train_with_max_vocab_size():
        """Test training with a maximum vocabulary size."""
        tokenizer = BPETokenizer()
        max_vocab_size = 30
        tokenizer.train(sample_texts(), max_vocab_size=max_vocab_size)
        assert len(tokenizer.vocab) == max_vocab_size, "Tokenizer did not limit vocabulary size correctly"


def test_tokenize():
    """Test tokenizing texts"""
    tokenizer = BPETokenizer()
    tokenizer.merges = [('t', 'h'), ('h', 'e'), ('n', 'g'), ('i', 'ng'), ('c', 'k'), ('th', 'e')]

    # trained_tokenizer.train(sample_texts(), max_merges=5)
    texts = ["the quick learning", 'hello word']
    tokenized = tokenizer.tokenize(texts)
    expected = [['the', 'q', 'u', 'i', 'ck', 'l', 'e', 'a', 'r', 'n', 'ing'], 
                ['he', 'l', 'l', 'o', 'w', 'o', 'r', 'd']]
    assert tokenized == expected, f"Tokenization failed: {tokenized} != {expected}"

#################################################
if __name__ == "__main__":
    import pytest
    import sys
    test_result = pytest.main(['--tb=short', __file__])
    if test_result != 0:
        sys.exit(test_result)
    print("Great! All tests passed!")
    print("Training the tokenizer on the IMDB dataset...")

    main(num_merges=10000, max_vocab_size=40000) # TODO: change the number of merges and/or vocab size
    print("Tokenizer trained and saved successfully.")
