"""
zero_to_shakespeare_explained.py

Train a tiny character-level LSTM on Shakespeare text
and save the trained weights + vocabulary mapping.

Every block is commented to remind you what is going on.
"""

# --- IMPORTS ---

import torch              # PyTorch = library for tensors + automatic differentiation
import torch.nn as nn     # nn = neural network building blocks (layers, loss functions, etc.)
import torch.optim as optim  # optim = optimizers (Adam, SGD, etc.)
import requests           # To download the text file from the internet
import sys                # For system-level stuff (not really used here, but handy)
import random             # To pick random chunks of text during training
import time               # To time training
import pickle             # To save Python objects (like vocab dicts) to disk

# --- CONFIGURATION HYPERPARAMETERS ---
# These are knobs you can tweak. They are not learned by the model.

HIDDEN_SIZE = 256     # Size of the LSTM's "hidden state" vector (its internal memory dimension)
NUM_LAYERS = 2        # How many LSTM layers stacked on top of each other
LR = 0.002            # Learning rate (how big a step gradient descent takes each update)
SEQ_LEN = 100         # How many characters per training sequence
EPOCHS = 5000         # How many training iterations
PRINT_EVERY = 250     # How often we print loss
TEMP = 0.8            # Temperature for sampling in generate() (not used in training loop)
SAVE_PATH = "genesis_brain.pth"   # Where we save model weights
VOCAB_PATH = "genesis_vocab.pkl"  # Where we save the character ↔ index mappings

# --- STEP 1: THE DATA ---

print("--- LOADING REALITY ---")
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

try:
    # Download the Shakespeare text as one long string
    text = requests.get(url).text
    # Trim it to 200,000 characters to keep training fast
    text = text[:200000]  # 200k chars is enough for a vibe check
except Exception:
    # Fallback text if there is no internet
    print("Bruh. No internet? Using fallback.")
    text = "To be, or not to be, that is the question..." * 1000

# Build the list of unique characters that appear in the text
chars = sorted(list(set(text)))   # e.g. ['\n', ' ', '!', '"', ...]
data_size, vocab_size = len(text), len(chars)

print(f"Data: {data_size} chars | Vocab: {vocab_size} unique")

# --- BUILD CHARACTER ↔ INDEX MAPPINGS ---
# We need to turn characters into integer IDs so the model can work with them.
# This is basically building a "dictionary" for our tiny language.

char_to_ix = {ch: i for i, ch in enumerate(chars)}  # 'a' -> 0, 'b' -> 1, ...
ix_to_char = {i: ch for i, ch in enumerate(chars)}  # 0 -> 'a', 1 -> 'b', ...

# Save the vocabulary so the generation script can use the same mapping.
# If we change this mapping, old weights become useless (hence "Rosetta Stone").

with open(VOCAB_PATH, 'wb') as f:
    # pickle dumps Python objects to a file in a binary format
    pickle.dump({'char_to_ix': char_to_ix, 'ix_to_char': ix_to_char}, f)

print(f"Vocabulary saved to {VOCAB_PATH}")

def to_tensor(string: str) -> torch.Tensor:
    """
    Turn a Python string into a 1D tensor of integer character indices.

    Example: "Hi" -> [index_of_H, index_of_i]

    This is basically: for each character, look up its index in char_to_ix.
    """
    tensor = torch.zeros(len(string), dtype=torch.long)  # long = integer type
    for c in range(len(string)):
        tensor[c] = char_to_ix[string[c]]
    return tensor

# --- STEP 2: THE MODEL ---

class TinyBrain(nn.Module):
    """
    A tiny character-level language model using:
    - Embedding layer: map character IDs to vectors
    - LSTM: process sequence of vectors with memory
    - Linear decoder: map hidden state back to character logits

    High-level math idea:
    - We learn a function f(text_so_far) ≈ probability distribution over next character.
    - Training nudges weights so true next chars get higher probability.
    """

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        # Initialize parent class
        super(TinyBrain, self).__init__()

        # Embedding: turns char index (0..vocab_size-1) into a hidden_size-dimensional vector
        # Think "vector meaning" of each character, learned automatically.
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # LSTM: a recurrent layer that reads a sequence of embeddings.
        # It keeps an internal state (hidden + cell) that updates at each time-step.
        # Internally it’s just a bunch of matrix multiplications and nonlinearities.
        self.rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True  # input shape: (batch, seq_len, features)
        )

        # Decoder: takes each hidden state vector and turns it into scores (logits) for each character.
        # Later we apply softmax (inside the loss) to get probabilities.
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        """
        x: tensor of shape (batch, seq_len) with character indices.
        hidden: optional (h, c) for LSTM, where each is (num_layers, batch, hidden_size).

        Returns:
          - out: unnormalized scores for each character at each time-step
          - hidden: last hidden state, so we can continue sequence later
        """
        # Look up embeddings for each character ID
        embed = self.embedding(x)  # shape: (batch, seq_len, hidden_size)

        # Run the sequence through the LSTM
        out, hidden = self.rnn(embed, hidden)  # out: (batch, seq_len, hidden_size)

        # Map hidden states to logits over the vocabulary
        out = self.decoder(out)  # (batch, seq_len, vocab_size)

        return out, hidden

    def generate(self, start_str='A', predict_len=100, temperature=0.8):
        """
        Generate text one character at a time, starting from start_str.

        temperature:
          - <1.0  → more conservative, less random (picks high-probability chars)
          - >1.0  → more chaotic, more random
        """
        hidden = None

        # Turn the starting text into tensor indices
        inp = to_tensor(start_str).unsqueeze(0)  # shape: (1, len(start_str))

        predicted = start_str

        # "Warm up" the hidden state by feeding all but the last character
        for p in range(len(start_str) - 1):
            _, hidden = self.forward(inp[:, p:p+1], hidden)

        # Now keep only the last character as input
        inp = inp[:, -1:]  # shape: (1, 1)

        # Generate predict_len new characters
        for _ in range(predict_len):
            # Forward pass: get logits for the next character
            output, hidden = self.forward(inp, hidden)

            # output shape is (1, 1, vocab_size) -> flatten to (vocab_size,)
            logits = output.data.view(-1)

            # Scale logits by temperature, then exponentiate to get unnormalized probabilities
            # Mathematically: probs ∝ exp(logits / temperature)
            output_dist = logits.div(temperature).exp()

            # Sample a character index according to this probability distribution
            top_i = torch.multinomial(output_dist, 1)[0]

            # Convert index back to character
            predicted_char = ix_to_char[top_i.item()]
            predicted += predicted_char

            # New input is the character we just generated
            inp = to_tensor(predicted_char).unsqueeze(0)

        return predicted


# Create the model with our vocab size and hyperparameters
model = TinyBrain(vocab_size, HIDDEN_SIZE, NUM_LAYERS)

# Adam optimizer:
# This is a fancier gradient descent method that adapts the step size per parameter.
optimizer = optim.Adam(model.parameters(), lr=LR)

# CrossEntropyLoss:
# Combines softmax + negative log-likelihood.
# It measures how "wrong" our predicted probabilities are for the true next char.
criterion = nn.CrossEntropyLoss()

# --- STEP 3: THE GRIND (TRAINING LOOP) ---

print("--- TRAINING STARTED ---")
start_time = time.time()
avg_loss = 0.0

for epoch in range(1, EPOCHS + 1):
    # Pick a random starting position in the text
    # We sample random chunks instead of going sequentially for simplicity.
    start_index = random.randint(0, data_size - SEQ_LEN - 1)

    # Take SEQ_LEN+1 characters:
    # first SEQ_LEN are inputs, last SEQ_LEN are targets (shifted by 1)
    chunk = text[start_index:start_index + SEQ_LEN + 1]

    # Turn characters into tensors of indices
    # input_seq: all chars except last
    # target_seq: all chars except first (we want to predict each next char)
    input_seq = to_tensor(chunk[:-1]).unsqueeze(0)   # (1, SEQ_LEN)
    target_seq = to_tensor(chunk[1:]).unsqueeze(0)   # (1, SEQ_LEN)

    # Reset gradients to zero before backprop
    optimizer.zero_grad()

    # Forward pass: get predicted logits for each time-step
    output, _ = model(input_seq, None)  # output: (1, SEQ_LEN, vocab_size)

    # Reshape to (batch * seq_len, vocab_size) to feed into CrossEntropyLoss
    loss = criterion(
        output.view(-1, vocab_size),       # predicted scores
        target_seq.view(-1)                # true next-char indices
    )

    # Backpropagation:
    # Autograd computes gradients of loss w.r.t. all model parameters.
    loss.backward()

    # Clip gradients to avoid them blowing up (stabilizes training)
    nn.utils.clip_grad_norm_(model.parameters(), 5)

    # Apply one optimization step: w := w - lr * gradient
    optimizer.step()

    # Track average loss for printing
    avg_loss += loss.item()

    # Occasionally print training progress
    if epoch % PRINT_EVERY == 0:
        print(f"Step {epoch}/{EPOCHS} | Loss: {avg_loss / PRINT_EVERY:.4f}")
        avg_loss = 0.0

# --- STEP 4: SAVE THE BRAIN ---

print("--- SAVING SIGMA STATE ---")
# Save only the weights (state_dict), not the whole model object
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model weights saved to {SAVE_PATH}")
print("Run 'load_brain_explained.py' (or your own loader) to hear it speak.")
