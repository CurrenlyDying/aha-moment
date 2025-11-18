# Aha Moment — TinyBrain Shakespeare

## Quick Demo: 4 Commands for the “Aha!” Moment

On Windows (PowerShell or CMD):

```bash
python .\explaimed\zero_to_shakespeare_explained.py
python .\explaimed\load_brain_explained.py
python .\explaimed\pipeline_visualizer.py
python .\explaimed\tinybrain_turbo_stream.py
````

On macOS / Linux:

```bash
python3 ./explaimed/zero_to_shakespeare_explained.py
python3 ./explaimed/load_brain_explained.py
python3 ./explaimed/pipeline_visualizer.py
python3 ./explaimed/tinybrain_turbo_stream.py
```

Run them in order to go from **training** → **first samples** → **seeing the pipeline** → **full-speed infinite generation**.

---

## What This Repo Is

A tiny character-level LSTM trained on Shakespeare text that’s designed to give beginners an **“oh wow, that’s all it takes?”** moment:
* 29 lines to make, save and train the model.
* Way more lines to allow the user to understand how to repoduce and use the art of modeling.
* A few hundred lines of Python to shoff visualizations and give insights on the data piping.
* Some mysteriously picked parameters.
* And your terminal starts speaking Shakespeare-ish text in real time

There are two parallel versions of the code:

* `explaimed/` — **fully commented**, educational versions
* `no_comment/` — **minified**, no-comments versions to show how small the core really is

---

## File / Folder Structure

```text
aha-moment/
  LICENSE
  README.md
  explaimed/
    zero_to_shakespeare_explained.py
    load_brain_explained.py
    pipeline_visualizer.py
    tinybrain_turbo_stream.py
  no_comment/
    zero_to_shakespeare.py
    load_brain.py
    pipeline_visualizer.py
    tinybrain_turbo_stream.py
```

### `explaimed/` (commented, teaching versions)

* **`zero_to_shakespeare_explained.py`**

  * Downloads the Tiny Shakespeare dataset
  * Builds the vocab file (`genesis_vocab.pkl`)
  * Trains a small LSTM (`genesis_brain.pth`)
  * Logs loss + sample generations over time, with “training stage” labels
#####  The author wishes to apologize in advance for the deliberately misleading labels. They love lying.
* **`load_brain_explained.py`**

  * Loads the saved model + vocab
  * Lets you type a prompt and streams text back with a “typing” effect
  * Good for showing the first “it’s alive” moment

* **`pipeline_visualizer.py`**

  * Loads the model + vocab
  * Takes a prompt and walks through the pipeline slowly:
    `text → indices → tensor → embeddings → LSTM → logits → probabilities → next char`
  * Prints shapes and a few values so you can see how data moves through the network

* **`tinybrain_turbo_stream.py`**

  * Loads the model + vocab
  * Asks for a prompt once
  * Then generates **infinite text at max speed** in your terminal until you hit `Ctrl+C`
  * This is the “models are insanely fast at inference” showcase

### `no_comment/` (minified versions)

* **`zero_to_shakespeare.py`**
* **`load_brain.py`**
* **`pipeline_visualizer.py`**
* **`tinybrain_turbo_stream.py`**

These are functionally the same as the `explaimed/` scripts but stripped of comments and extra whitespace to highlight how little code you *actually* need.

---

## Requirements

* Python 3.8+
* [PyTorch](https://pytorch.org/) (CPU is fine for this demo)
* `requests` (for downloading the dataset)

Install dependencies (just replace with any missing package if needed):

```bash
pip install torch requests
```

Then run the 4 commands at the top in order and enjoy watching a tiny LSTM wake up, learn, and then stream Shakespeare-ish text at ridiculous speed.

