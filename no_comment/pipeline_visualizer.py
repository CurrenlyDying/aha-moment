import torch, torch.nn as nn
import pickle, sys, time, os, random

MODEL_PATH = "genesis_brain.pth"
VOCAB_PATH = "genesis_vocab.pkl"
HIDDEN_SIZE = 256
NUM_LAYERS = 2
SPEED = 0.03  # text speed for explanations

def slow_print(text="", delay=SPEED):
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()

def section(title):
    slow_print("\n" + "=" * 60)
    slow_print(f"{title}")
    slow_print("=" * 60 + "\n")

if not os.path.exists(VOCAB_PATH):
    print(f"Missing {VOCAB_PATH}. Run the training script first.")
    sys.exit()

if not os.path.exists(MODEL_PATH):
    print(f"Missing {MODEL_PATH}. Run the training script first.")
    sys.exit()

with open(VOCAB_PATH, "rb") as f:
    vocab_data = pickle.load(f)
    char_to_ix = vocab_data["char_to_ix"]
    ix_to_char = vocab_data["ix_to_char"]

vocab_size = len(char_to_ix)

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

model = TinyBrain(vocab_size, HIDDEN_SIZE, NUM_LAYERS)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

def to_tensor(s):
    t = torch.zeros(len(s), dtype=torch.long)
    for i, ch in enumerate(s):
        t[i] = char_to_ix.get(ch, 0)
    return t

def visualize_once(prompt, temperature=0.8):
    section("STEP 1: RAW TEXT")
    slow_print(f'Prompt: "{prompt}"')

    section("STEP 2: CHARACTERS → INDICES (using vocab file)")
    idxs = []
    for ch in prompt:
        idx = char_to_ix.get(ch, 0)
        safe = ch if ch != "\n" else "\\n"
        slow_print(f"  '{safe}'  →  {idx}")
        idxs.append(idx)
        time.sleep(0.2)

    idx_tensor = torch.tensor(idxs, dtype=torch.long).unsqueeze(0)
    slow_print(f"\nTensor shape: {tuple(idx_tensor.shape)} (batch, seq_len)")
    slow_print(f"Tensor data: {idx_tensor.tolist()}")

    with torch.no_grad():
        section("STEP 3: INDICES → EMBEDDINGS")
        emb = model.embedding(idx_tensor)
        slow_print(f"Embeddings shape: {tuple(emb.shape)} (batch, seq_len, hidden_size)")
        slow_print("First token embedding (truncated):")
        first_vec = emb[0, 0, :8]  # show first 8 dims
        slow_print("  " + " ".join(f"{v.item():+.2f}" for v in first_vec))

        section("STEP 4: EMBEDDINGS → LSTM → HIDDEN STATES")
        out, hidden = model.rnn(emb, None)
        slow_print(f"LSTM output shape: {tuple(out.shape)} (batch, seq_len, hidden_size)")
        slow_print("Hidden state shape: "
                   f"h={tuple(hidden[0].shape)}, c={tuple(hidden[1].shape)}")

        section("STEP 5: HIDDEN STATES → LOGITS OVER VOCAB")
        logits = model.decoder(out)  # (batch, seq_len, vocab_size)
        slow_print(f"Logits shape: {tuple(logits.shape)} (batch, seq_len, vocab_size)")

        last_logits = logits[0, -1]  # last time step
        probs = torch.softmax(last_logits / temperature, dim=-1)

        slow_print("\nNext-char probabilities (top 5):")
        top_p, top_i = torch.topk(probs, 5)
        for p, i in zip(top_p, top_i):
            ch = ix_to_char[i.item()]
            safe = ch if ch != "\n" else "\\n"
            slow_print(f"  '{safe}'  ~  {p.item():.3f}")
            time.sleep(0.2)

        section("STEP 6: SAMPLE ONE NEXT CHARACTER")
        sampled_idx = torch.multinomial(probs, 1)[0].item()
        sampled_char = ix_to_char[sampled_idx]
        safe = sampled_char if sampled_char != "\n" else "\\n"
        slow_print(f"Sampled index: {sampled_idx}")
        slow_print(f"Sampled character: '{safe}'")
        slow_print("\nSo the pipeline was:")
        slow_print("  text → indices → embeddings → LSTM → logits → probabilities → next char")
        slow_print("All powered by:")
        slow_print("  • vocab file (mapping chars ↔ indices)")
        slow_print("  • model file (weights for embedding, LSTM, decoder)")

@torch.no_grad()
def quick_stream(prompt, length=120, temperature=0.8):
    section("BONUS: WATCH THE PIPELINE LOOP IN REAL TIME")
    slow_print("Now the same pipeline runs in a loop to write text.\n")
    h = None
    inp = to_tensor(prompt).unsqueeze(0)

    for ch in prompt:
        sys.stdout.write(f"\033[92m{ch}\033[0m")
        sys.stdout.flush()
        time.sleep(random.uniform(0.02, 0.06))

    for i in range(len(prompt) - 1):
        _, h = model(inp[:, i:i+1], h)
    inp = inp[:, -1:]

    for _ in range(length):
        o, h = model(inp, h)
        logits = o.view(-1)
        probs = torch.softmax(logits / temperature, dim=-1)
        idx = torch.multinomial(probs, 1)[0].item()
        ch = ix_to_char[idx]
        sys.stdout.write(f"\033[92m{ch}\033[0m")
        sys.stdout.flush()
        time.sleep(random.uniform(0.02, 0.08))
        inp = torch.tensor([[idx]], dtype=torch.long)
    sys.stdout.write("\n")

if __name__ == "__main__":
    print("=" * 60)
    print(" TINYBRAIN PIPELINE VISUALIZER")
    print("=" * 60)
    try:
        prompt = input('Enter a short prompt (default: "The King"): ').strip()
    except KeyboardInterrupt:
        print("\nBye.");sys.exit()
    if not prompt:
        prompt = "The King"
    visualize_once(prompt, temperature=0.8)
    quick_stream(prompt, temperature=0.8)
