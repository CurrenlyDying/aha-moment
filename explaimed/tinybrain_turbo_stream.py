import torch, torch.nn as nn
import pickle, sys, os

MODEL_PATH = "genesis_brain.pth"
VOCAB_PATH = "genesis_vocab.pkl"
HIDDEN_SIZE = 256
NUM_LAYERS = 2

# --- SAFETY CHECKS: FILES MUST EXIST ---

if not os.path.exists(VOCAB_PATH):
    print(f"Missing {VOCAB_PATH}. Run the training script first.")
    sys.exit()

if not os.path.exists(MODEL_PATH):
    print(f"Missing {MODEL_PATH}. Run the training script first.")
    sys.exit()

# --- LOAD VOCABULARY (CHAR ↔ INDEX) ---

with open(VOCAB_PATH, "rb") as f:
    vocab_data = pickle.load(f)
    char_to_ix = vocab_data["char_to_ix"]
    ix_to_char = vocab_data["ix_to_char"]

vocab_size = len(char_to_ix)

# --- MODEL DEFINITION (MUST MATCH TRAINING) ---

class TinyBrain(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        e = self.embedding(x)
        o, h = self.rnn(e, h)
        o = self.decoder(o)
        return o, h

# --- INSTANTIATE + LOAD WEIGHTS ---

model = TinyBrain(vocab_size, HIDDEN_SIZE, NUM_LAYERS)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# --- SMALL HELPER: STRING → TENSOR OF INDICES ---

def to_tensor(s):
    t = torch.zeros(len(s), dtype=torch.long)
    for i, ch in enumerate(s):
        t[i] = char_to_ix.get(ch, 0)
    return t.unsqueeze(0)  # shape: (1, seq_len)

# --- TURBO INFERENCE LOOP: INFINITE STREAM, NO SLEEPS ---

@torch.no_grad()
def turbo_stream(prompt="The King", temperature=0.8):
    """
    Infinite text stream at max speed.
    Stop with Ctrl+C.
    """
    # Convert prompt to indices
    inp = to_tensor(prompt)
    h = None

    # Warm up hidden state on all but the last character
    for i in range(len(prompt) - 1):
        _, h = model(inp[:, i:i+1], h)

    # Keep only the last character as current input
    inp = inp[:, -1:]

    # Print the initial prompt in green (ANSI), then go infinite
    for ch in prompt:
        sys.stdout.write(f"\033[92m{ch}\033[0m")
    sys.stdout.flush()

    # Infinite generation loop
    while True:
        o, h = model(inp, h)
        logits = o.view(-1)
        probs = torch.softmax(logits / temperature, dim=-1)
        idx = torch.multinomial(probs, 1)[0].item()
        ch = ix_to_char[idx]

        # Write char (green) as fast as possible, no sleep
        sys.stdout.write(f"\033[92m{ch}\033[0m")
        sys.stdout.flush()

        # Next input is just the sampled char
        inp = torch.tensor([[idx]], dtype=torch.long)

if __name__ == "__main__":
    print("=" * 60)
    print(" TINYBRAIN TURBO STREAM")
    print(" (Ctrl+C to stop)")
    print("=" * 60)
    try:
        prompt = input('Enter a short prompt (default: "The King"): ').strip()
    except KeyboardInterrupt:
        print("\nBye.")
        sys.exit()
    if not prompt:
        prompt = "The King"

    try:
        turbo_stream(prompt=prompt, temperature=0.8)
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Turbo stream stopped.")
