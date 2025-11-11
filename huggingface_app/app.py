# app.py — Gradio UI with Inference, Validation, 📈 Dashboard (+ Compare runs),
# ONNX export, and Hub-hosted weights via hf_hub_download.

# ---- Environment fixes (before importing NumPy/Torch) ----
import os
os.environ["OMP_NUM_THREADS"] = "1"  # silence libgomp warning in HF Spaces

import io, json, glob
import numpy as np
import pandas as pd
from PIL import Image
import gradio as gr
import torch, torch.nn as nn
from torchvision import models, transforms, datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import hf_hub_download

# ---- Local utils (Grad-CAM) ----
from cam_utils import grad_cam

# ===================== Config =====================
MODEL_NAME  = os.environ.get("MODEL_NAME", "efficientnet_b0")
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "2"))
IMAGE_SIZE  = int(os.environ.get("IMAGE_SIZE", "224"))
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Parasitized", "Uninfected"] if NUM_CLASSES == 2 else [str(i) for i in range(NUM_CLASSES)]

# Prefer Hub → local env path → checkpoints/* fallback
HF_REPO_ID   = os.environ.get("HF_REPO_ID", "").strip()
HF_WEIGHTS   = os.environ.get("HF_WEIGHTS", "best.pt").strip() if HF_REPO_ID else ""
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "checkpoints/best.pt")


def resolve_weights() -> str:
    # 1) Try Hub
    if HF_REPO_ID:
        try:
            path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_WEIGHTS)
            return path
        except Exception as e:
            print(f"[hub] failed to download {HF_REPO_ID}:{HF_WEIGHTS} → {e}")
    # 2) Local explicit env path
    if os.path.exists(WEIGHTS_PATH):
        return WEIGHTS_PATH
    # 3) Local fallbacks
    candidates = ["checkpoints/best.pt", "checkpoints/last.pt"] + sorted(glob.glob("checkpoints/*.pt"))
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No checkpoint found. Upload a .pt into /checkpoints, "
        "or set HF_REPO_ID/HF_WEIGHTS, or WEIGHTS_PATH."
    )


# ===================== Model =====================
def build_model(name: str, num_classes: int):
    name = name.lower()
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unsupported model_name: {name}")


CHECKPOINT_FILE = resolve_weights()
_model = build_model(MODEL_NAME, NUM_CLASSES).to(DEVICE)
_state = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
try:
    _model.load_state_dict(_state)               # plain state_dict
except Exception:
    _model.load_state_dict(_state["state_dict"]) # or checkpoint dict
_model.eval()

_pre = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE*1.15)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
])


# ===================== Inference =====================
def predict(image: Image.Image, show_cam: bool):
    if image is None:
        return {"label": "", "confidences": []}, None
    img = image.convert("RGB")
    x = _pre(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = _model(x).cpu().numpy().squeeze()
    probs = np.exp(logits - logits.max()); probs = probs / probs.sum()

    # Gradio Label expects dict {class: prob}
    label_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    overlay = None
    if show_cam:
        cam = grad_cam(_model, img, img_size=IMAGE_SIZE, device=DEVICE)
        overlay = Image.fromarray((cam["overlay"]*255).astype("uint8"))
    return label_dict, overlay


# ===================== ONNX Export =====================
def export_onnx(precision: str):
    m = build_model(MODEL_NAME, NUM_CLASSES).to(DEVICE)
    state = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    try:
        m.load_state_dict(state)
    except Exception:
        m.load_state_dict(state["state_dict"])
    m.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    if precision == "fp16":
        m = m.half(); dummy = dummy.half()

    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
    buf = io.BytesIO()
    torch.onnx.export(
        m, dummy, buf,
        input_names=["input"], output_names=["output"],
        dynamic_axes=dynamic_axes, opset_version=17, do_constant_folding=True
    )
    fname = f"model_{MODEL_NAME}_{precision}_{IMAGE_SIZE}.onnx"
    buf.seek(0)
    # Gradio File expects (filename, filelike/bytes)
    return fname, buf


# ===================== Validation (optional) =====================
def validate(zip_file):
    import tempfile, zipfile
    if zip_file is None:
        return "Upload a .zip of your validation set.", None

    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file.name, 'r') as zf:
        zf.extractall(tmp)

    ds = datasets.ImageFolder(tmp, transform=_pre)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in dl:
            preds = _model(xb.to(DEVICE)).argmax(1).cpu().numpy()
            ys.extend(yb.numpy()); ps.extend(preds)

    import sklearn.metrics as sk
    rep = sk.classification_report(ys, ps, target_names=ds.classes, output_dict=True)
    cm  = sk.confusion_matrix(ys, ps)

    fig, ax = plt.subplots(figsize=(4.5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=ds.classes, yticklabels=ds.classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=160); buf.seek(0)
    cm_img = Image.open(buf).convert("RGB")

    return json.dumps(rep, indent=2), cm_img


# ===================== Dashboard helpers =====================
METRICS_DEFAULT = "checkpoints/metrics.csv"

def _plot_to_pil(df: pd.DataFrame, ycol: str, title: str, ylabel: str):
    if ycol not in df.columns: return None
    s = df[["epoch", ycol]].dropna()
    if len(s) == 0: return None
    fig, ax = plt.subplots(figsize=(5.6,3.2))
    ax.plot(s["epoch"], s[ycol], marker="o")
    ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def load_metrics(path: str):
    if not os.path.exists(path):
        return "No metrics file found.", None, None, None, None, None, None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return f"Failed to read CSV: {e}", None, None, None, None, None, None

    tail = df.tail(10).to_markdown(index=False)
    fig_loss = _plot_to_pil(df, "train_loss", "Training Loss", "loss")
    fig_acc  = _plot_to_pil(df, "val_acc",   "Validation Accuracy", "acc")
    fig_act  = _plot_to_pil(df, "act_rate",  "Activation Rate", "fraction")
    fig_save = _plot_to_pil(df, "save_rate", "Energy Savings", "fraction")
    fig_thr  = _plot_to_pil(df, "threshold", "Activation Threshold", "threshold")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    # Return a tuple (filename, bytes) for DownloadButton.value
    return tail, fig_loss, fig_acc, fig_act, fig_save, fig_thr, ("metrics.csv", csv_bytes)

def upload_metrics(file):
    if file is None:
        return "No file uploaded.", None, None, None, None, None, None
    os.makedirs("checkpoints", exist_ok=True)
    dst = METRICS_DEFAULT
    with open(dst, "wb") as f:
        f.write(file.read())
    return load_metrics(dst)

# ---- Compare runs: overlay multiple metrics.csv files ----
def _overlay_from_runs(runs, ycol, title, ylabel):
    fig, ax = plt.subplots(figsize=(6.2,3.2))
    found = False
    for name, df in runs:
        if ycol in df.columns:
            s = df[["epoch", ycol]].dropna()
            if len(s):
                ax.plot(s["epoch"], s[ycol], marker="o", label=name)
                found = True
    if not found:
        plt.close(fig)
        return None
    ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=160, bbox_inches="tight"); buf.seek(0)
    return Image.open(buf).convert("RGB")

def compare_runs(files):
    if not files or len(files) == 0:
        return "Upload 2+ metrics.csv files to compare.", None, None, None, None, None
    runs = []
    for f in files:
        try:
            df = pd.read_csv(f.name)
            runs.append((os.path.basename(f.name), df))
        except Exception as e:
            return f"Failed reading {f.name}: {e}", None, None, None, None, None

    msg    = f"Compared {len(runs)} runs."
    p_loss = _overlay_from_runs(runs, "train_loss", "Training Loss (overlay)", "loss")
    p_acc  = _overlay_from_runs(runs, "val_acc",    "Validation Accuracy (overlay)", "acc")
    p_act  = _overlay_from_runs(runs, "act_rate",   "Activation Rate (overlay)", "fraction")
    p_save = _overlay_from_runs(runs, "save_rate",  "Energy Savings (overlay)", "fraction")
    p_thr  = _overlay_from_runs(runs, "threshold",  "Activation Threshold (overlay)", "threshold")
    return msg, p_loss, p_acc, p_act, p_save, p_thr


# ===================== Gradio UI =====================
with gr.Blocks(title="Malaria Diagnostic Assistant") as demo:
    gr.Markdown("# 🩸 Malaria Diagnostic Assistant")
    gr.Markdown("Prototype — energy-efficient triage with human-in-the-loop (Adaptive Sparse Training).")

    # -------- Inference --------
    with gr.Tab("🔍 Inference"):
        gr.Markdown(f"**Weights:** `{os.path.basename(CHECKPOINT_FILE)}` • **Backbone:** `{MODEL_NAME}` • **Device:** `{DEVICE}`")
        with gr.Row():
            with gr.Column(scale=1):
                img_in   = gr.Image(type="pil", label="Upload blood smear image")
                show_cam = gr.Checkbox(value=True, label="Show Grad-CAM")
                btn_pred = gr.Button("Predict", variant="primary")

                gr.Markdown("### Export ONNX")
                onnx_precision = gr.Radio(choices=["fp32", "fp16"], value="fp32", label="Precision")
                btn_onnx = gr.Button("Export ONNX")
                onnx_file = gr.File(label="Download ONNX", interactive=False)

            with gr.Column(scale=1):
                label_out = gr.Label(num_top_classes=2, label="Prediction & Probabilities")
                cam_out   = gr.Image(type="pil", label="Grad-CAM overlay")

        btn_pred.click(fn=predict, inputs=[img_in, show_cam], outputs=[label_out, cam_out])

        def onnx_wrap(prec):
            fname, fobj = export_onnx(prec)
            # File component expects (filename, filelike/bytes)
            return (fname, fobj)

        btn_onnx.click(fn=onnx_wrap, inputs=[onnx_precision], outputs=[onnx_file])

    # -------- Validation --------
    with gr.Tab("✅ Validation (optional)"):
        gr.Markdown("Upload a **.zip** containing a folder with class subfolders (e.g., `Parasitized/`, `Uninfected/`).")
        val_zip   = gr.File(label="Validation ZIP", file_types=[".zip"])
        btn_eval  = gr.Button("Compute report + confusion matrix")
        rep_out   = gr.Textbox(label="classification_report (JSON)")
        cm_img    = gr.Image(type="pil", label="Confusion Matrix")
        btn_eval.click(fn=validate, inputs=[val_zip], outputs=[rep_out, cm_img])

    # -------- Dashboard --------
    with gr.Tab("📈 Dashboard"):
        gr.Markdown("Visualize training logs from `checkpoints/metrics.csv` (written each epoch by your Colab training).")

        with gr.Row():
            metrics_path = gr.Textbox(value=METRICS_DEFAULT, label="Metrics CSV path")
            btn_load     = gr.Button("Refresh")

        tail_md   = gr.Markdown(value="Upload metrics or click refresh.")
        plot_loss = gr.Image(label="Training Loss")
        plot_acc  = gr.Image(label="Validation Accuracy")
        plot_act  = gr.Image(label="Activation Rate")
        plot_save = gr.Image(label="Energy Savings")
        plot_thr  = gr.Image(label="Activation Threshold")
        dl_btn    = gr.DownloadButton(label="⬇️ Download metrics.csv", value=None)

        btn_load.click(
            fn=load_metrics,
            inputs=[metrics_path],
            outputs=[tail_md, plot_loss, plot_acc, plot_act, plot_save, plot_thr, dl_btn]
        )

        gr.Markdown("---")
        gr.Markdown("**Compare runs** — upload multiple `metrics.csv` files to overlay curves:")
        mult = gr.Files(label="Upload one or more metrics.csv files", file_types=[".csv"])
        cmp_msg = gr.Markdown()
        with gr.Row():
            p_loss = gr.Image(label="Loss (overlay)")
            p_acc  = gr.Image(label="Val Acc (overlay)")
        with gr.Row():
            p_act  = gr.Image(label="Activation (overlay)")
            p_save = gr.Image(label="Savings (overlay)")
        p_thr  = gr.Image(label="Threshold (overlay)")

        mult.upload(
            fn=compare_runs,
            inputs=[mult],
            outputs=[cmp_msg, p_loss, p_acc, p_act, p_save, p_thr]
        )

    gr.Markdown("---")
    gr.Markdown("Built with EfficientNet-B0 + Adaptive Sparse Training — not a diagnostic device.")

if __name__ == "__main__":
    demo.launch()
