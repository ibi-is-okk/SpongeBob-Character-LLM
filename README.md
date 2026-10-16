# SpongeBob Character LLM & Chat UI

LoRA fine-tune of an open-source LLM (default `unsloth/Llama-3.2-3B-Instruct`) to speak like SpongeBob, plus a themed Streamlit chat app with streaming replies.

## 1. Project Purpose & Non-Commercial Disclaimer
This repository is an **educational, non-commercial research and learning project**. It exists to demonstrate LoRA fine-tuning, persona alignment, and building a chat UI. It is a parody/simulation, **not affiliated with, endorsed by, or sponsored by Nickelodeon or Paramount**. SpongeBob SquarePants and related names are trademarks of their owners.

Please note:
- **No transcripts, model weights, or artwork are included.** Show transcripts are copyrighted. You must supply data you have the right to use, and you should not redistribute the dataset or the resulting weights.
- Do not publish, monetize, or publicly deploy the model or app.
- The UI shows the disclaimer banner: *"Educational AI Simulation / Parody Project — Not Affiliated with Nickelodeon."* Keep it.
- Background: drop your own image at `assets/background.jpg` (or `.png`) for personal use. Without one, the app uses a sky-to-seabed gradient.

## 2. Upstream Vision / Future Integration
This character module is designed as the **core dialogue engine for a broader real-time multimodal agent project**. Planned scope:
- A vision/screen-perception layer (periodic screenshots or a screen stream fed to a vision-language model) so SpongeBob can continuously "see" the user's computer screen.
- Real-time commentary and interaction on what the user is doing, with the perception output injected into the prompt as context.
- Later additions: speech in/out and low-latency streaming.

The `SYSTEM_PROMPT` in `training/persona.py` and the adapter's chat template are the stable interface that the future agent will call.

## 3. Repository Layout
```
training/data_cleaner.py     transcripts -> JSONL (instruction, response)
training/train.py            Unsloth + LoRA fine-tune
training/train_colab.ipynb   Colab notebook wrapping the steps below
training/persona.py          shared system prompt (training + inference)
app/app.py                   Streamlit chat UI
data/raw/                    your transcripts (git-ignored)
models/                      LoRA adapter output (git-ignored)
assets/                      optional background.jpg (git-ignored)
```

## 4. Setup & Execution

### Step 1: Fine-tune on Google Colab (T4)
1. Push this repo to your GitHub, then open `training/train_colab.ipynb` in Colab. Set the runtime to **T4 GPU**.
2. Edit the clone URL in cell 2, then run the cells in order: install, upload your `.txt` transcripts (`Speaker: line` format), clean the data, train.

Or run the commands directly in a Colab terminal/cell:
```bash
pip install unsloth
python training/data_cleaner.py --input data/raw --output data/processed/spongebob.jsonl
python training/train.py --data data/processed/spongebob.jsonl --out models/spongebob-lora --epochs 2
```
Tip: for Mistral use `--base unsloth/mistral-7b-v0.3-bnb-4bit` and reduce `--max-seq` to 768 if you run out of memory.

### Step 2: Download the LoRA weights
The adapter is small (tens of MB). The notebook zips and downloads `models/spongebob-lora`. Or:
```bash
zip -r spongebob-lora.zip models/spongebob-lora   # then download from the Colab file browser
```
Unzip into `models/spongebob-lora/` in your local clone.

### Step 3: Install requirements
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
An NVIDIA GPU is recommended. Without CUDA the app falls back to a slow CPU path (it pulls the full base model, so expect large downloads). On Colab, install Unsloth with `pip install unsloth` as in the notebook rather than via `requirements.txt`.

### Step 4: Launch the web UI
```bash
streamlit run app/app.py
# custom adapter location:
ADAPTER_PATH=/path/to/adapter streamlit run app/app.py
```
Open http://localhost:8501. To run it inside Colab, use the last notebook cell (ngrok tunnel).


## 4b. No NVIDIA GPU? Train on your laptop CPU/RAM
Uses a small model (SmolLM2-360M, about 4-8 GB RAM). Slower and rougher than the Colab route, but fully local.
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-cpu.txt
python training/data_cleaner.py --input data/raw --output data/processed/spongebob.jsonl
python training/train_cpu.py                          # caps at 1500 examples, 1 epoch
streamlit run app/app.py
```
Smoke test without transcripts: `--input data/sample` runs the pipeline end to end, but 6 pairs are far too few to learn a real style.
If it runs out of memory, add `--max-seq 128 --max-samples 500`. If it is too slow, lower `--max-samples`.

## 5. Persona Alignment
Persona comes from two layers: the LoRA weights (speech patterns learned from your dialogue pairs) and the light system prompt in `training/persona.py` (lore, laugh, catchphrases). The same prompt is used for training and inference, so change it in one place only and retrain if you alter it substantially.

## 6. Troubleshooting
- **CUDA out of memory:** lower `--max-seq`, or use the 3B model.
- **Few training pairs:** check that speaker names match `--character` (default `SpongeBob`).
- **Version conflicts:** Unsloth moves fast; if installs clash, install Unsloth first and let it choose compatible `transformers`/`trl`.
