"""
load_brain_explained.py

Load the trained TinyBrain model + vocabulary
and stream generated Shakespeare-like text to the console.

Comments explain what each piece is doing, in beginner-friendly language.
"""

import torch             # Tensors + neural nets
import torch.nn as nn    # Layers, etc.
import pickle            # To load the saved vocabulary dicts
import sys               # For exiting cleanly
import time              # For typing effect delays
import random            # To randomize typing speed a bit
import os                # To check if files exist

# --- CONFIGURATION ---

MODEL_PATH = "genesis_brain.pth"   # Weights saved from the training script
VOCAB_PATH = "genesis_vocab.pkl"   # Vocabulary mapping saved earlier
HIDDEN_SIZE = 256                  # Must match training config
NUM_LAYERS = 2                     # Must match training config
TEMPERATURE = 0.6                  # Lower = more serious / predictable. Higher = more chaotic.
SPEED = 0.05                       # Base typing speed (seconds per char)

# --- STEP 1: REVIVE VOCABULARY ---

# Check that the vocabulary file exists
if not os.path.exists(VOCAB_PATH):
    print(f"Bruh. Missing {VOCAB_PATH}. Did you run the training script?")
    sys.exit()

# Load vocab data (char_to_ix and ix_to_char) from disk
with open(VOCAB_PATH, 'rb') as f:
    vocab_data = pickle.load(f)

char_to_ix = vocab_data['char_to_ix']  # char -> index
ix_to_char = vocab_data['ix_to_char']  # index -> char

vocab_size = len(char_to_ix)
print(f"--- VOCAB LOADED ({vocab_size} chars) ---")

def to_tensor(string: str) -> torch.Tensor:
    """
    Turn a string into a tensor of indices using char_to_ix.

    We use .get(..., 0) so unknown characters default to index 0
    instead of crashing. (Not perfect, but simple.)
    """
    tensor = torch.zeros(len(string), dtype=torch.long)
    for c in range(len(string)):
        tensor[c] = char_to_ix.get(string[c], 0)
    return tensor

# --- STEP 2: RECONSTRUCT THE BRAIN ---

# The model structure must match EXACTLY what we used for training.
# Same layers, same sizes, same order. Otherwise the saved weights won't fit.

class TinyBrain(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        super(TinyBrain, self).__init__()

        # Same embedding as in training
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Same LSTM config as in training
        self.rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True
        )

        # Same decoder layer
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        """
        Forward pass for one or more time-steps.

        x: (batch, seq_len) of indices
        hidden: (h, c) hidden state for LSTM or None

        Returns logits over characters at each time-step + final hidden state.
        """
        embed = self.embedding(x)
        out, hidden = self.rnn(embed, hidden)
        out = self.decoder(out)
        return out, hidden


# Create a fresh instance of the model
model = TinyBrain(vocab_size, HIDDEN_SIZE, NUM_LAYERS)

# Load the saved weights into this model
try:
    # torch.load loads the state dict from disk
    state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict)

    # eval() tells PyTorch we're not training anymore (turns off dropout, etc.)
    model.eval()
    print("--- SIGMA WEIGHTS LOADED ---")
except Exception as e:
    print(f"Cringe. Could not load model: {e}")
    sys.exit()

# --- STEP 3: THE STREAMING GENERATOR ---

def stream_generation(start_str: str, length: int = 500, temp: float = 0.8):
    """
    Generate text starting from start_str and stream it character by character
    with a typing effect.

    Internally:
      - We repeatedly run one time-step of the model.
      - The model outputs logits (scores) over characters.
      - We use temperature-scaled softmax to sample the next character.
    """
    hidden = None

    # Convert the prompt to indices
    inp = to_tensor(start_str).unsqueeze(0)  # shape: (1, len(start_str))

    # Print the starting string in green, one character at a time
    for char in start_str:
        # \033[92m and \033[0m are ANSI codes for green text and reset.
        sys.stdout.write(f"\033[92m{char}\033[0m")
        sys.stdout.flush()
        time.sleep(random.uniform(SPEED * 0.5, SPEED * 1.5))

    # Use the model to "warm up" its hidden state on the prompt
    with torch.no_grad():  # Turn off gradient tracking (we're not training)

        # Feed all but the last character to build up the hidden state
        for p in range(len(start_str) - 1):
            _, hidden = model(inp[:, p:p+1], hidden)

        # Keep only the last character as the current input
        inp = inp[:, -1:]  # shape: (1, 1)

        # Now generate new characters one by one
        for _ in range(length):
            # Forward pass for the current character
            output, hidden = model(inp, hidden)  # output: (1, 1, vocab_size)

            # Flatten logits to (vocab_size,)
            logits = output.data.view(-1)

            # Temperature scaling: logits / temp, then exponentiate
            # Mathematically: probs ∝ exp(logits / temp)
            output_dist = logits.div(temp).exp()

            # Sample one index according to this distribution
            top_i = torch.multinomial(output_dist, 1)[0]

            # Convert index back to a character
            char = ix_to_char[top_i.item()]

            # Print the character in green
            sys.stdout.write(f"\033[92m{char}\033[0m")
            sys.stdout.flush()

            # Typing effect: pause depending on the character
            delay = SPEED
            if char in ['.', ',', '!', '?', ':']:
                delay *= 4   # Dramatic pause at punctuation
            elif char == ' ':
                delay *= 1.5 # Slightly longer after spaces
            elif char == '\n':
                delay *= 3   # Longer pause at line breaks

            # Add a bit of randomness so it feels less robotic
            time.sleep(random.uniform(delay * 0.5, delay * 1.5))

            # New input is the character we just generated
            inp = to_tensor(char).unsqueeze(0)

    # Finish with a newline so the shell prompt looks normal
    sys.stdout.write("\n")


# --- MAIN LOOP ---

while True:
    print("\n" + "=" * 40)
    try:
        # Ask user for a prompt
        prompt = input("\033[94mGive the Oracle a prompt (or Enter for default): \033[0m")
    except KeyboardInterrupt:
        # Allow Ctrl+C to exit cleanly
        print("\nExiting matrix...")
        break

    # Default prompt if user just hits Enter
    if not prompt:
        prompt = "The King"

    print("-" * 20)

    # Generate ~400 characters using the global TEMPERATURE setting
    stream_generation(prompt, length=400, temp=TEMPERATURE)
