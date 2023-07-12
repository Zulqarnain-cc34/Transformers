import torch.nn as nn
import torch
import math

# First of all we need to convert each word in the input sequence to an embedding vector.
# Embedding vectors will create a more semantic representation of each word.

# Suppose each embedding vector is of 512 dimension and suppose our vocab size is 100,
# then our embedding matrix will be of size 100x512. These marix will be learned on 
# training and during inference each word will be mapped to corresponding 512 d vector. 
# Suppose we have batch size of 32 and sequence length of 10(10 words). The the output will 
# be 32x10x512.

# In the embedding matrix:

# 1. Each row represents a word in the vocabulary.
# 2. The number of rows corresponds to the vocab size.
# 3. The number of columns represents the dimensionality of the embedding vectors (eg: 512)

class Embedding(nn.Module):
    """ 
    A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices. 
    The input to the module is a list of indices, and the output is the corresponding word embeddings

    """
    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        output = self.embed(x)
        return output

# Next step is to generate positional encoding. Inorder for the model to make # sense of the 
# sentence, it needs to know two things about the each word.

# 1. what does the word mean?
# 2. what is the position of the word in the sentence.

# The model itself doesn’t have any sense of position/order for each word.
# Consequently, there’s still the need for a way to incorporate the order of the words into our model.

# Position and order of words are the essential parts of any language.
# Ideally our position embedding should satisfy this criteria

# 1. It should output a unique encoding for each time-step (word’s position in a sentence)
# 2. Distance between any two time-steps should be consistent across sentences with different lengths.
# 3. Our model should generalize to longer sentences without any efforts. Its values should be bounded.
# 4. It must be deterministic.

# Proposed method
# ---------------

# The encoding proposed by the authors is a simple yet genius technique which satisfies all of those 
# criteria. First of all, it isn’t a single number. 

# Instead, it’s a d-dimensional vector that contains 
# information about a specific position in a sentence. And secondly, this encoding is not integrated 
# into the model itself. Instead, this vector is used to equip each word with information about its 
# position in a sentence. In other words, we enhance the model’s input to inject the order of words.


#  ```
# pos -> refers to order in the sentence
# i -> refers to position along embedding vector dimension
# ```


class PositionalEmbedding(nn.Module):
    """
    Positional embedding will generate a matrix of similar to embedding matrix. It will create a matrix of 
    dimension sequence length x embedding dimension.For each token(word) in sequence, we will find the embedding
    vector which is of dimension 1 x 512 and it is added with the correspondng positional vector which is of 
    dimension 1 x 512 to get 1 x 512 dim out for each word/token.

    for eg: if we have batch size of 32 and seq length of 10 and let embedding dimension be 512. Then we will have
    embedding vector of dimension 32 x 10 x 512. Similarly we will have positional encoding vector of dimension 
    32 x 10 x 512. Then we add both.

    """
    def __init__(self, max_seq_len,embed_model_dim):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()

        # To store the embedding dimension (eg: 512)
        self.embed_dim = embed_model_dim

        # Making a embeddings matrix of (seq_len x embed_dim) shape which is zero to begin with
        pe = torch.zeros(max_seq_len, self.embed_dim)

        # looping over the sentence
        for pos in range(max_seq_len):
            # looping over the word
            # We are skipping 1 loop since we are doing that in the odd and even step below
            # We are updating both cosine and sine alternating it in even and odd
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))

        # The positional encoding matrix pe is unsqueezed along the first dimension 
        # to add a batch dimension and stored as a buffer using register_buffer
        # Let say initially the shape of vector was lets say (100,512) it turns it 
        # into (1,100,512) to be able to add batches to it as well
        # self.pe = pe.unsqueeze(0)
        self.pe = pe
        # self.register_buffer('pe', self.pe)


    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        return x
