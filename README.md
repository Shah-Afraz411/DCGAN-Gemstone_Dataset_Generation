# Synthetic Image Generation and Identification of Gemstones
> DCGAN-based pipeline for generating realistic gemstone images and (optionally) improving gemstone identification through synthetic data augmentation.

<p align="center">
  <img src="results/samples/sample_epoch_200.png" alt="Generated Gemstone Samples (Epoch 200)" width="640"/>
</p>

---

## 🧪 Project Goals
1. Train a Deep Convolutional GAN (DCGAN) to synthesize diverse, high-quality gemstone images.
2. Use generated images to augment a (future/optional) gemstone classification model.
3. Establish a reproducible preprocessing + training + evaluation workflow.
4. Track improvements using quantitative (FID, IS) and qualitative (visual grid) metrics.

---

## 📂 Repository Contents
| Path | Description |
|------|-------------|
| `dcgan-gemstone-dataset-generation.ipynb` | End‑to‑end notebook: preprocessing → model definition → training → sample generation |
| `README.md` | Project documentation (this file) |
| `results/` *(expected)* | Training artifacts: generated image grids, checkpoints, metrics JSON |
| `data/` *(expected)* | Raw and processed gemstone images |
| `models/` *(optional)* | Saved `.pt` or `.pth` files |

> Only the notebook exists currently. Other directories are created automatically or when you follow the setup steps.

---

## 🧷 Key Features
- Clean DCGAN implementation in PyTorch
- Modular hyperparameter block (latent dim, feature maps, learning rate)
- Deterministic seeds for reproducibility
- Progressive sample logging by epoch
- (Planned) FID + Inception Score calculation
- (Planned) Identification model fine-tuning with synthetic augmentation

---

## 🏞 Dataset
You can use:
- A curated gemstone image dataset (your own photography or public domain sources)
- Public datasets (ensure licensing compliance)

Recommended minimum: ≥ 1,000 real images across gemstone classes (e.g., ruby, emerald, sapphire, topaz, amethyst).  
Store raw images under:
```
data/
  raw/
    ruby/*.jpg
    emerald/*.jpg
    ...
```

---

## 🔄 Preprocessing Pipeline
| Step | Purpose | Typical Setting |
|------|---------|-----------------|
| Load & Validate | Filter unreadable / corrupt files | `PIL.Image.open()` try/except |
| Color Space | Ensure 3-channel consistency | Convert to RGB |
| Resize | Stabilize receptive fields | `resize → 64x64` (or 128x128 if dataset size supports) |
| Center / Smart Crop | Remove borders / frames | Center crop square |
| Augment (Optional for Discriminator only) | Increase variability | Random horizontal flip (p=0.5) |
| Tensor Conversion | PyTorch format | `ToTensor()` |
| Normalization | Help stable GAN training | Normalize to `[-1, 1]` via `(x - 0.5)/0.5` |
| Class Index Mapping (for future ID task) | Supervised augmentation | `class_to_idx.json` |

Example PyTorch transform:
```python
transform = transforms.Compose([
    transforms.Resize(72),
    transforms.CenterCrop(64),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
```

---

## 🧱 Model Architecture (DCGAN)

### Generator (G)
- Input: latent vector `z ∈ R^{Z}` (default `Z=100`)
- Architecture: Transposed Conv blocks (stride=2) + BatchNorm + ReLU
- Output: `3 x 64 x 64` image with `tanh` activation

| Layer | Output Shape | Notes |
|-------|--------------|-------|
| Linear/Reshape | `ngf*8 x 4 x 4` | Start feature map |
| ConvT Block 1 | `ngf*4 x 8 x 8` | BatchNorm + ReLU |
| ConvT Block 2 | `ngf*2 x 16 x 16` | BatchNorm + ReLU |
| ConvT Block 3 | `ngf x 32 x 32` | BatchNorm + ReLU |
| ConvT Block 4 | `3 x 64 x 64` | Tanh |

### Discriminator (D)
- Input: `3 x 64 x 64`
- Conv → LeakyReLU (0.2) → (optional) SpectralNorm → final sigmoid (or raw logits + BCEWithLogitsLoss)

| Layer | Output Shape | Notes |
|-------|--------------|-------|
| Conv 1 | `ndf x 32 x 32` | No BatchNorm (DCGAN best practice) |
| Conv 2 | `ndf*2 x 16 x 16` | BatchNorm |
| Conv 3 | `ndf*4 x 8 x 8` | BatchNorm |
| Conv 4 | `ndf*8 x 4 x 4` | BatchNorm |
| Linear | `1` | Real/Fake score |

---

## ⚙️ Training Configuration (Typical Defaults)
| Hyperparameter | Value |
|----------------|-------|
| Image Size | 64 |
| Latent Dim (Z) | 100 |
| Batch Size | 128 |
| Optimizer | Adam (`β1=0.5, β2=0.999`) |
| LR | 0.0002 |
| Epochs | 200 (tune as needed) |
| Loss | Binary Cross Entropy |
| Weight Init | Normal (`mean=0, std=0.02`) |
| Checkpoint Freq | Every 10 epochs |

---

## 🚀 Quick Start

### 1. Clone
```bash
git clone https://github.com/Shah-Afraz411/DCGAN-Gemstone_Dataset_Generation.git
cd DCGAN-Gemstone_Dataset_Generation
```

### 2. (Recommended) Create Environment
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows) .venv\Scripts\activate
pip install --upgrade pip
```

### 3. Install Dependencies (example)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pillow numpy matplotlib tqdm scipy scikit-image
```

### 4. Prepare Data
Place images under `data/raw/<class_name>/` then modify the dataset root inside the notebook.

### 5. Run the Notebook
Open and execute:
```
dcgan-gemstone-dataset-generation.ipynb
```

---

## 🧪 Evaluation Metrics (Planned / Recommended)
| Metric | Why It Matters | Tooling |
|--------|----------------|---------|
| FID (Fréchet Inception Distance) | Measures realism & diversity | `torch-fidelity` |
| Inception Score | Diversity + recognizability | `pytorch-gan-metrics` |
| Precision / Recall (Generative) | Mode dropping vs overfitting | `gen-eval` libs |
| Per-Class Augmentation Impact | For identification model | Downstream classifier accuracy |

Example (placeholder) results table:

| Epoch | FID ↓ | Inception Score ↑ | Notes |
|-------|-------|-------------------|-------|
| 50 | 78.4 | 4.15 | Faces forming, color blotchy |
| 100 | 56.2 | 4.78 | Edges sharpen |
| 150 | 44.9 | 5.02 | Better hue separation |
| 200 | 39.3 | 5.11 | Saturation stabilized |

> Replace with actual numbers once computed.

---

## 🖼 Sample Outputs
(Insert generated grids periodically saved during training.)
```
results/
  samples/
    sample_epoch_050.png
    sample_epoch_100.png
    sample_epoch_150.png
    sample_epoch_200.png
```

Embed like:
```markdown
| Epoch 50 | Epoch 100 | Epoch 150 | Epoch 200 |
|----------|-----------|-----------|-----------|
| ![](results/samples/sample_epoch_050.png) | ![](results/samples/sample_epoch_100.png) | ![](results/samples/sample_epoch_150.png) | ![](results/samples/sample_epoch_200.png) |
```

---

## 🧭 Identification (Optional Extension)
After generating synthetic images:
1. Split real data: 70% train / 15% val / 15% test.
2. Generate N synthetic images per class (balance underrepresented classes).
3. Train a classifier (e.g., ResNet18) on:
   - Scenario A: Real only
   - Scenario B: Real + Synthetic
4. Compare accuracy / F1 / per-class recall.

Potential pipeline snippet:
```python
train_real = load_real_dataset(...)
synthetic = load_generated_images(...)
augmented = ConcatDataset([train_real, synthetic])
```

---

## 🛠 Improvements / Experiments
| Technique | Rationale |
|-----------|-----------|
| Label Smoothing (e.g., real=0.9) | Stabilizes D |
| Spectral Normalization | Controls Lipschitz constant |
| TTUR (different G/D LRs) | Converges more reliably |
| Gradient Penalty (WGAN-GP) | Better gradients, fewer artifacts |
| Minibatch Std Dev Layer | Boosts diversity |
| Diffusion Model Baseline | Compare generative quality (future) |

---

## 🧯 Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| Checkerboard artifacts | Kernel/stride mismatch | Use kernel=4, stride=2, pad=1 |
| Mode collapse | Overtrained D | Lower D LR or add noise |
| Saturated loss (~0 or ~1) | Imbalance G/D | Apply gradient penalty or freeze D steps |
| Washed-out colors | Improper normalization | Ensure `[-1,1]` pipeline consistent |
| FID plateau | Limited data diversity | Add augmentations or gather more real images |

---

## 🗺 Roadmap
- [ ] Add metrics script (`evaluate.py`)
- [ ] CLI training script (decouple from notebook)
- [ ] Classification augmentation experiments
- [ ] 128×128 upscale experiment
- [ ] Diffusion baseline comparison
- [ ] Model card + ethical usage note

---

## 🤝 Contributing
1. Fork & create a feature branch
2. Follow style (PEP8, docstrings)
3. Add sample results (if visual changes)
4. Open PR with clear description

---

## 📄 License
No license specified yet.  
Add a `LICENSE` file (e.g., MIT or Apache-2.0) if you intend others to reuse.

---

## 🧾 Citation (Template)
If you use this repository in academic or research work:

```bibtex
@software{synthetic_gemstone_dcgan_2025,
  author = {Syed Afraz},
  title = {Synthetic Image Generation and Identification of Gemstones},
  year = {2024},
  url = {https://github.com/Shah-Afraz411/DCGAN-Gemstone_Dataset_Generation}
}
```

---

## 🙏 Acknowledgements
- DCGAN paper: Radford, Metz, Chintala (2015)
- PyTorch examples: Official GAN tutorials
- Community resources on generative model stabilization

---

## ⚠️ Ethical / Responsible Use
Synthetic images should not be used to misrepresent the authenticity of gemstones in commercial contexts. Always disclose augmentation in research or product pipelines.

---

## 📬 Contact
Open an issue or PR for suggestions, improvements, or result sharing.

---

### ✅ Next Steps for You
Update:
- Real metric values
- Actual sample images
- Add evaluation script
- Add license

Happy generating! 💎

**Author:** Shah Afraz  
**Repository:** [DCGAN-Gemstone_Dataset_Generation](https://github.com/Shah-Afraz411/DCGAN-Gemstone_Dataset_Generation)
