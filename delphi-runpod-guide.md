# Gemma 3 Transcoder Feature Labels — RunPod Setup Guide

A step-by-step guide to running the Delphi interpretability pipeline on a RunPod GPU instance.

---

## 1. Connect to Your Pod

```bash
ssh <your-pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

---

## 2. Clone & Install

```bash
cd /workspace
git clone https://github.com/AutoInterp/gemma3-transcoder-feature-labels.git
cd gemma3-transcoder-feature-labels

python -m venv venv
source venv/bin/activate

pip install -e .
pip install plotly
```

---

## 3. Set Hugging Face Tokens

```bash
export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxx
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx
```

---

## 4. Configure Persistent Cache (RunPod-Specific)

RunPod's `/root` directory is ephemeral — only `/workspace` persists across restarts. This step ensures model weights and datasets are stored on the persistent volume.

```bash
# Create persistent cache directories
mkdir -p /workspace/.cache/huggingface/datasets

# Set environment variables permanently
echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets' >> ~/.bashrc
source ~/.bashrc

# Symlink so anything hardcoded to /root/.cache still works
rm -rf /root/.cache/huggingface
ln -sf /workspace/.cache/huggingface /root/.cache/huggingface

# Verify
echo "HF_HOME=$HF_HOME"
ls -la /root/.cache/huggingface/
```

---

## 5. Set Up Git (Optional)

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

---

## 6. Run Delphi in a tmux Session

Using tmux ensures the process keeps running even if your SSH connection drops.

```bash
apt update && apt install -y tmux
tmux new -s delphi
```

Inside the tmux session:

```bash
cd /workspace/gemma3-transcoder-feature-labels
source venv/bin/activate

python -m delphi \
  google/gemma-3-4b-it \
  mwhanna/gemma-scope-2-4b-it/transcoder_all/width_16k_l0_small_affine \
  --n_tokens 10_000_000 \
  --max_latents 16384 \
  --hookpoints 15 \
  --explainer_provider offline \
  --explainer_model "unsloth/Meta-Llama-3.1-8B-Instruct" \
  --explainer_model_max_len 5120 \
  --name gemma3_4b_it_layer15
```

Change `--hookpoints` and `--name` to match your target layer (e.g., `14` and `gemma3_4b_it_layer14`).

---

## 7. Build Feature Labels Dictionary

After Delphi finishes, generate the labels JSON:

```bash
python build_labels_dict.py \
  --explanations_dir results/gemma3_4b_it_layer15/explanations \
  --output feature_labels.json \
  --model unsloth/Meta-Llama-3.1-8B-Instruct
```

---

## 8. tmux Cheat Sheet

| Action | Command |
|---|---|
| Detach from session | `Ctrl + B` then `D` |
| Reattach to session | `tmux attach -t delphi` |
| New window | `Ctrl + B` then `C` |
| Switch to window 0 | `Ctrl + B` then `0` |
| Kill stuck process | `Ctrl + C` (or `pkill -9 -f "python -m delphi"` from another window) |

---

## Troubleshooting

**`safetensors_rust.SafetensorError: incomplete metadata`**
A model file was partially downloaded. Delete the corrupted cache and re-run:

```bash
rm -rf /workspace/.cache/huggingface/hub/models--google--gemma-3-4b-it
# Then re-run the delphi command
```

**`NameError: name 'cache' is not defined`**
Check for version mismatches between Delphi and vLLM. Try updating:

```bash
pip install -e . --upgrade
```

**Process won't stop with `Ctrl + C`**
Open a new terminal, SSH in again, and force kill:

```bash
pkill -9 -f "python -m delphi"
```
