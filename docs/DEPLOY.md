# Deploying to Hugging Face Spaces

A Space gives the app a permanent URL instead of a `gradio.live` link that expires after a week.

1. **Upload the weights to a model repo** (keeps the Space repo small):
   ```bash
   pip install huggingface_hub
   hf auth login
   hf repo create <user>/dental-xray-yolov9 --type model
   hf upload <user>/dental-xray-yolov9 weights/best.pt best.pt
   ```
2. **Create a Gradio Space** at <https://huggingface.co/new-space> (SDK: Gradio; CPU basic works, a GPU is faster).
3. **Push this repository to the Space.** The Space's `README.md` needs a front-matter block; create the Space with
   this as its README (GitHub keeps the normal README):
   ```yaml
   ---
   title: Dental X-ray AI Analyzer
   emoji: 🦷
   sdk: gradio
   app_file: app.py
   pinned: false
   license: mit
   ---
   ```
4. **Set the Space's variables and secrets** (*Settings → Variables and secrets*):
   | Name | Type | Value |
   |---|---|---|
   | `HF_MODEL_REPO` | variable | `<user>/dental-xray-yolov9` |
   | `ANTHROPIC_API_KEY` | secret | optional, enables the Q&A tab |
   | `MIN_RELIABLE_AP50` / `MIN_TRAIN_INSTANCES` | variable | optional gating thresholds |

Do not set `SHARE` on a Space. A public Space is reachable by anyone: add a clear notice, keep it private,
or require login before letting people upload identifiable patient radiographs.
