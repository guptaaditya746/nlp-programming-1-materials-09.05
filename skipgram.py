#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List,  Tuple
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from bpe import BPETokenizer, load_imdb_dataset

GROUP = "27"  # TODO: write in your group number

class SkipGram(nn.Module):
    """
    Model for skipgram implementation with one-hot encoding.
    """
    def __init__(self, vocab_size: int, vector_dim: int) -> None:
        """
        Initialize the Skip-gram model.
        Args:
            vocab_size: Size of the vocabulary
            vector_dim: Dimension of the token embeddings
        """
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.vector_dim = vector_dim
        
        self.input_to_hidden = nn.Linear(vocab_size, vector_dim) # target embeddings
        self.hidden_to_output = nn.Linear(vector_dim, vocab_size) # context embeddings
        self.output_probs = nn.LogSoftmax(dim=1) # probabilities of each token in 
                                                 # the vocabulary to be the context token
    

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.
        Takes a batch of input token indices, encodes them to one-hot vectors,
        and computes the log probabilities of the vocabulary.
        Args:
            inputs: Input token indices (batch_size)
        Returns:
            Log probabilities for each token in the vocabulary
        """
        # NLP:III-81
        # TODO: Your code here 
        # convert input indices to one-hot vectors
        one_hot_inputs = self._to_one_hot(inputs)
        hidden = self.input_to_hidden(one_hot_inputs)
        output = self.hidden_to_output(hidden)
        log_probs = self.output_probs(output)
        # project one hot vectors to hidden layer (lookup step)

        # project to output layer and
        # compute log softmax probabilities (forward computation)
        return log_probs
    

    def _to_one_hot(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert token indices to one-hot vectors (NLP:III-53).
        Takes a tensor of indices and returns a matrix of 
        one-hot vectors of size [batch_size, vocab_size].
        Args:
            indices: Tensor of token indices [batch_size]
        Returns:
            One-hot encoded vectors [batch_size, vocab_size]
        """
        # TODO: Your code here
        batch_size = indices.size(0)
        one_hot = torch.zeros(batch_size, self.vocab_size)
        one_hot.scatter_(1, indices.unsqueeze(1),1.0)
        return one_hot
    

    def get_token_embedding(self, token_idx: int) -> torch.Tensor:
        """
        Get the embedding for a specific token index.
        Args:
            token_idx: Index of the token
        Returns:
            token embedding
        """
        # TODO: Your code here
        # encode the token index to a one-hot vector
        # return the hidden layer output
        token_idx_tensor = torch.tensor([token_idx], dtype=torch.long)
        one_hot_vector = self._to_one_hot(token_idx_tensor)
        # return the hidden layer output
        embedding = self.input_to_hidden(one_hot_vector)
        return embedding.squeeze(0)




class SkipGramTrainer:
    """
    Trainer for Skip-gram model using BPE tokenization.
    """
    def __init__(self, vector_dim=100, context_size=2, learning_rate=0.01, epochs=5):
        """
        Initialize the skipgram trainer.
        
        Args:
            vector_dim: Dimension of token embeddings
            context_size: Size of the context window
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
        """
        self.vector_dim = vector_dim
        self.context_size = context_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.tokenizer =  BPETokenizer().load(path=f"./output/bpe_group_{GROUP}.json")
        self.token2id = self.tokenizer.vocab if self.tokenizer else None
        self.id2token = {v: k for k, v in self.token2id.items()} if self.token2id else None
        self.vocab_size = len(self.token2id) if self.token2id else 0

        self.model = SkipGram(self.vocab_size, self.vector_dim)
        

    def _generate_context_pairs(self, tokenized_corpus: List[List[str]]) -> List[Tuple[int, int]]:
        """
        Generate training data for Skip-gram model.
        Args:
            tokenized_corpus: List of tokenized documents
        Returns:
            List of (target_token, context_token) pairs within the context size
        """
        # TODO: Your code here
        # for each token in the document, create context pairs
        # context pairs are tuples of (target token id, context token id)
        # context tokens are within the context size
        context_pairs = []
        for doc_words in tqdm(tokenized_corpus, desc="Generating context pairs"):
            # If tokenized_corpus is List[List[str]], then doc_words is already
            # a flat list of tokens for the current document.
            doc_tokens = doc_words 
            
            for i, target_token in enumerate(doc_tokens):
                target_id = self.token2id.get(target_token)
                if target_id is None:
                    continue # Skip if target token not in vocab

                # Define the context window boundaries
                start = max(0, i - self.context_size)
                end = min(len(doc_tokens), i + self.context_size + 1)

                for j in range(start, end):
                    if i == j: # Target token cannot be its own context
                        continue
                    context_token = doc_tokens[j]
                    context_id = self.token2id.get(context_token)
                    if context_id is not None: # Ensure context token is in vocab
                        context_pairs.append((target_id, context_id))
        return context_pairs 
    
        
    def train(self, texts: List[str], batch_size: int=512) -> None:
        """
        Train the Skip-gram model on the given texts.
        
        Args:
            texts: List of text documents
        """
        # this method was already implemented for you
        print("Tokenizing texts...")
        tokenized_texts = self.tokenizer.tokenize(texts) # UPD.: 09.05
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_function = nn.NLLLoss()
        
        print("Generating training data...")
        training_data = self._generate_context_pairs(tokenized_texts)
        print(f"Generated {len(training_data)} training examples")
        
        print("Training model...")
        losses = []
        
        for epoch in range(self.epochs):
            total_loss = 0
            random.shuffle(training_data)
            num_batches = len(training_data) // batch_size
            # batch gradient descent (ML:III-136)
            for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.epochs}"):
                batch = training_data[i*batch_size:(i+1)*batch_size]
                target_ids = torch.tensor([pair[0] for pair in batch], dtype=torch.long)
                context_ids = torch.tensor([pair[1] for pair in batch], dtype=torch.long)
                self.model.zero_grad()
                # forward pass
                log_probs = self.model(target_ids)
                loss = loss_function(log_probs, context_ids)
                # backpropagation
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.epochs+1), losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('training_loss.png')
        print("Training complete!")
    
    def cosine_similarity(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        """
        NLP:III-114
        Calculate cosine similarity between two vectors.
        
        Args:
            v2: First vector
            v2: Second vector
        Returns:
            Cosine similarity between v1 and v2
        """
        # TODO: Your code here
        dot_product = torch.dot(v1, v2)
        norm_v1 = torch.norm(v1)
        norm_v2 = torch.norm(v2)
        return (dot_product / (norm_v1 * norm_v2 + 1e-8)).item() 
            

    def find_similar_tokens(self, token, top_n=10):
        """
        Find the most similar tokens to the given token using cosine similarity.
        
        Args:
            token: The token to find similar tokens for
            top_n: Number of similar tokens to return
            
        Returns:
            List of (token, cosine similarity score) tuples
        """
        # TODO: Your code here
        # hint: use the get_token_embedding method to get the embedding of the token
        # and then compare it to all other tokens in the vocabulary (token2id keys)
        if token not in self.token2id:
            raise ValueError(f"Token '{token}' not found in vocabulary.")

        target_token_id = self.token2id[token]
        target_embedding = self.model.get_token_embedding(target_token_id).detach() # Detach from computation graph

        similarities = []
        for other_token, other_token_id in self.token2id.items():
            if other_token == token:
                continue # Skip comparing the token with itself
            
            other_embedding = self.model.get_token_embedding(other_token_id).detach()
            similarity = self.cosine_similarity(target_embedding, other_embedding)
            similarities.append((other_token, similarity))

        # Sort by similarity score in descending order
        similarities.sort(key=lambda item: item[1], reverse=True)
        
        return similarities[:top_n]
    

    def save(self, path=f'./output/skipgram_group_{GROUP}.pt'):
        """
        Save the trained model and vocabulary.
        
        Args:
            path: Path to save the model
        """
        # This method was already implemented for you
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        save_data = {
            'model_state': self.model.state_dict(),
            'id2token': self.id2token,
            'token2id': self.token2id,
            'vector_dim': self.vector_dim,
            'context_size': self.context_size
        }
        
        torch.save(save_data, path)
        print(f"Model saved to {path}")
        

    @classmethod
    def load(cls, path=f'./output/skipgram_group_{GROUP}.pt'):
        """
        Load a trained model.
        Args:
            path: Path to load the model from
            tokenizer_path: Path to load the tokenizer from
        Returns:
            Loaded SkipGramTrainer instance
        """
        # this method was already implemented for you
        save_data = torch.load(path)

        trainer = cls(
            vector_dim=save_data['vector_dim'],
            context_size=save_data['context_size']
        )
        
        trainer.id2token = save_data['id2token']
        trainer.token2id = save_data['token2id']
        trainer.vocab_size = len(trainer.id2token)
        
        trainer.model = SkipGram(trainer.vocab_size, trainer.vector_dim)
        trainer.model.load_state_dict(save_data['model_state'])
        
        print(f"Model loaded from {path}")
        return trainer

def main(vector_dim: int, context_size: int, 
         learning_rate: float, epochs: int, small_dataset: bool = False):
    """
    Main function to load the dataset, train the model, and save it.
    
    Args:
        data_proportion: Proportion of the dataset to load (in %)
        vector_dim: Dimension of token embeddings
        context_size: Size of the context window
        learning_rate: Learning rate for optimization
        epochs: Number of training epochs
    """
    print("Loading IMDB dataset...")
    dataset = load_imdb_dataset(small_dataset=small_dataset)
    
    w2v = SkipGramTrainer(vector_dim=vector_dim, context_size=context_size,
                          learning_rate=learning_rate, epochs=epochs)
    w2v.train(dataset)
    
    w2v.save(f'skipgram_group_{GROUP}.pt')

    test_samples = ['the', 'movie', 'story', 'is', 
                    'great', 'bad', 'good', 
                   'boring', 'interesting', 'funny',
                   'how', 'they', 'people', 'very', 'ed', 'ing', 
                     'ly', 'er', 'est', 'able', 'ible']
    with open("./output/skipgram_test_group_XX.txt", "w") as f:
        for token in test_samples:
            try:
                similar_tokens = w2v.find_similar_tokens(token, top_n=5)
                print(f"Most similar tokens to '{token}':")
                for sim_token, sim_value in similar_tokens:
                    print(f"{sim_token}: {sim_value:.4f}")
                    f.write(f"{token} {sim_token} {sim_value:.4f}\n")
                f.write("\n")
                print("\n")
            except:
                print(f"Token '{token}' not found in vocabulary.\n")
                f.write(f"Token '{token}' not found in vocabulary.\n\n")
    print("Results saved to output/skipgram_test_group_XX.txt")
    print("Done!")

########################################################################
# Tests
import pytest
import os
import sys

def test_that_the_group_name_is_there():
    import re
    assert re.match(r'^[0-9]{1,2}$', GROUP), \
        "Please write your group name in the variable at the top of the file!"
    
def test_to_one_hot():
    """Test the _to_one_hot method of the SkipGram model."""
    vocab_size = 8
    vector_dim = 10
    model = SkipGram(vocab_size, vector_dim)
    
    indices = torch.tensor([1, 3, 5])
    one_hot = model._to_one_hot(indices)
    print(one_hot)
    assert one_hot.shape == (3, 8)
    
    assert one_hot[0, 1].item() == 1.0
    assert one_hot[1, 3].item() == 1.0
    assert one_hot[2, 5].item() == 1.0

    assert torch.sum(one_hot[0]).item() == 1.0
    assert torch.sum(one_hot[1]).item() == 1.0
    assert torch.sum(one_hot[2]).item() == 1.0


def test_forward():
    """Test the forward method of the SkipGram model."""
    vocab_size = 8
    vector_dim = 10
    model = SkipGram(vocab_size, vector_dim)

    indices = torch.tensor([1, 3, 5])
    output = model(indices)
    assert output.shape == (3, 8)
    
    for i in range(output.shape[0]):
        assert abs(torch.exp(output[i]).sum().item() - 1.0) < 1e-5


def test_get_token_embedding():
    """Test the get_token_embedding method of the SkipGram model."""
    vocab_size = 8
    vector_dim = 10
    model = SkipGram(vocab_size, vector_dim)
    
    token_idx = 2
    embedding = model.get_token_embedding(token_idx)

    assert embedding.shape == (10,)
    
    one_hot = torch.zeros(1, 8)
    one_hot[0, token_idx] = 1.0
    expected_embedding = model.input_to_hidden(one_hot).detach().squeeze(0)
    assert torch.allclose(embedding, expected_embedding)



def test_generate_context_pairs(): # UPD.: 09.05
    """Test the _generate_context_pairs method of the SkipGramTrainer."""
    mock_tokenizer = {}
    mock_tokenizer['vocab'] = {
        "the": 0,
        "quick": 1,
        "brown": 2,
        "fox": 3,
        "jumps": 4,
        "over": 5,
        "lazy": 6,
        "dog": 7
    }

    expected_pairs = {
            1:        # context size = 1
            [(0, 1), (1, 0),  (1, 2), (2, 1), (2, 3), (3, 2),  
            (4, 5), (5, 4), (5, 0), (0, 5), (0, 6), (6, 0), (6, 7), (7, 6)],
            
            2:        # context size = 2
            [(0, 1), (0, 2), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 1), (3, 2), 
             (4, 5), (4, 0), (5, 4), (5, 0), (5, 6), (0, 4), (0, 5), (0, 6), (0, 7), (6, 5), (6, 0), (6, 7), (7, 0), (7, 6)],
            
            3: # context size = 3
            [(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2), 
            (4, 5), (4, 0), (4, 6), (5, 4), (5, 0), (5, 6), (5, 7), (0, 4), (0, 5), (0, 6), (0, 7), (6, 4), (6, 5), (6, 0), (6, 7), (7, 5), (7, 0), (7, 6)]
    }
    print(expected_pairs)
    for context_size in [1, 2, 3]:
        trainer = SkipGramTrainer(vector_dim=10, context_size=context_size)
        trainer.token2id = mock_tokenizer["vocab"]
        
        tokenized_corpus = [["the", "quick", "brown", "fox"],
                        ["jumps", "over", "the", "lazy", "dog"]]
        
        pairs = trainer._generate_context_pairs(tokenized_corpus)

        expected = expected_pairs[context_size]
        pairs.sort()
        expected.sort()
    
        assert len(pairs) == len(expected), \
            f"Failed to generate the correct number of context pairs for context size {context_size}"



def test_cosine_similarity():
    """Test the cosine_similarity method of the SkipGramTrainer."""
    trainer = SkipGramTrainer(vector_dim=10, context_size=1)
    v1 = torch.tensor([1.0, 2.0, 3.0])
    v2 = torch.tensor([4.0, 5.0, 6.0])
    
    similarity = trainer.cosine_similarity(v1, v2)
    expected = 0.9746318461970762
    assert abs(similarity - expected) < 1e-6


if __name__ == "__main__":
    import pytest
    import sys
    test_result = pytest.main(['--tb=short', __file__])
    if test_result != 0:
        sys.exit(test_result)
    print("Great! All tests passed!")
    print("Running the main function...")


    main(vector_dim=100, context_size=5, learning_rate=0.01, epochs=3, small_dataset=False) # TODO: update the hyperparameters