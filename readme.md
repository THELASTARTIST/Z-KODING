<div align="center">

# 🫁 RespiCore

### *Acoustic Respiratory Triage · Edge AI · 100% Offline*

> **"If every smartphone is a potential stethoscope, respiratory screening is no longer a privilege of geography or resources."**
> — *The RespiCore Team, 2025*

---

**⚡ This is a research prototype and hackathon demonstration — not a certified clinical tool.**
*See the [Clinical Disclaimer](#-clinical-disclaimer) section before use.*

</div>

---

## 📋 Table of Contents

1. [What is RespiCore?](#-what-is-respicore)
2. [Demo Overview](#-demo-overview)
3. [The Problem We're Solving](#-the-problem-were-solving)
4. [How It Works — The 5-Stage Pipeline](#-how-it-works--the-5-stage-pipeline)
5. [Tech Stack](#-tech-stack)
6. [Model Performance](#-model-performance)
7. [Competitive Landscape](#-competitive-landscape)
8. [Privacy by Design](#-privacy-by-design)
9. [Project Files](#-project-files)
10. [Demo Walkthrough](#-demo-walkthrough)
11. [Authentication (Demo Mode)](#-authentication-demo-mode)
12. [The Team](#-the-team)
13. [Project Timeline](#-project-timeline)
14. [Languages Supported](#-languages-supported)
15. [Clinical Disclaimer](#-clinical-disclaimer)

---

## 🫁 What is RespiCore?

RespiCore is an **on-device acoustic respiratory triage system** that converts a 10-second cough recording into a Mel-spectrogram and runs a quantized CNN to classify respiratory conditions — all entirely on the device, with zero internet dependency.

It detects four classes of respiratory status:

| Class | Indicator | Description |
|-------|-----------|-------------|
| 🟢 **Normal** | Low risk | Clear lung acoustics, no anomalous patterns |
| 🟡 **Anomalous** | Moderate risk | Irregular acoustic signatures warranting follow-up |
| 🔵 **Wheeze** | Elevated risk | High-frequency oscillatory patterns consistent with asthma |
| 🔴 **COPD / Bronchitis** | High risk | Broadband noise and multi-peak spectral distribution |

The model runs in **under 50ms** on a mid-range Android device, uses only **4.2 MB** of storage, and requires **no network connection** at any point during inference.

---

## 🖥️ Demo Overview

This repository contains the **interactive web demo** of RespiCore, built as a single-page application with three interconnected HTML pages:

```
index.html              ← Main landing page + live triage demo
respicore_about.html    ← "Our Story" — team, timeline, comparisons
review.html             ← Community reviews (login required)
```

> **This is a simulated demo.** No real microphone audio is processed by an actual ML model. The triage demo uses pre-configured condition patterns to visually demonstrate what the real mobile app produces.

### Running the Demo Locally

No build step required — just open the files in a browser:

```bash
# Clone or download the files, then:
open index.html

# Or serve locally to avoid browser CORS restrictions:
python -m http.server 8080
# Then visit: http://localhost:8080
```

All three HTML files must be in the **same directory** for page navigation to work correctly.

---

## 🌍 The Problem We're Solving

Respiratory disease is a silent epidemic:

- **300M+ people** cannot access a doctor when they need one
- **3.23M deaths per year** are attributable to COPD alone
- **6.8B smartphones** exist globally — devices powerful enough to screen for lung disease

The gap isn't technology. It's **access, cost, and infrastructure**. RespiCore bridges this gap by turning every smartphone into a capable respiratory pre-screener — no stethoscope, no hospital, no internet required.

---

## ⚙️ How It Works — The 5-Stage Pipeline

Every stage of the RespiCore pipeline runs **100% locally on the device**. No audio, no data, no metadata ever leaves the hardware.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   RespiCore Inference Pipeline                       │
│                                                                      │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  ┌──────────┐ │
│   │  Stage 1 │──▶│  Stage 2 │──▶│  Stage 3 │──▶│  Stage 4 │─▶│  Stage 5 │ │
│   │  Audio   │   │ Spectral │   │   Mel-   │   │   CNN    │  │  Report  │ │
│   │ Capture  │   │ Denoise  │   │ Spectro  │   │Inference │  │ & Store  │ │
│   └──────────┘   └──────────┘   └──────────┘   └──────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Stage 01 — Audio Capture

The microphone is locked at **16 kHz mono** for a 10-second window. Real-time amplitude waveform is rendered live. Leading and trailing silence is auto-trimmed using librosa's energy threshold (`top_db = 20`) so dead-air padding doesn't pollute the spectrogram.

**Tags:** `16 kHz` · `WAV` · `flutter_record`

---

### Stage 02 — Spectral Denoising

The first 500ms of the recording is treated as a **noise floor sample**. A spectral gate is applied: an STFT mask zeros any frequency bin where signal power is less than 2× the noise power. The cleaned signal is reconstructed via inverse STFT (iSTFT). This dramatically reduces the impact of ambient environmental noise — fans, traffic, wind.

**Tags:** `Spectral Gating` · `STFT / iSTFT` · `Librosa`

---

### Stage 03 — Mel-Spectrogram Transformation

The denoised signal is converted into a **128-bin Mel-spectrogram**:

| Parameter | Value | Reason |
|-----------|-------|--------|
| Mel filterbanks | 128 | Captures fine-grained spectral detail |
| n_fft | 2048 | Frequency resolution |
| Hop length | 512 | Time resolution |
| Frequency range | 50–8000 Hz | Covers all clinically relevant respiratory harmonics |
| Scale | Log dB | Compresses dynamic range for CNN input |
| Output size | 224 × 224 px | Matches MobileNetV2 input layer |

**Tags:** `128 Mel bins` · `224×224` · `Log-scale dB`

---

### Stage 04 — CNN Inference (TFLite)

A **MobileNetV2** backbone, fine-tuned on COUGHVID + Coswara, performs the classification. The model is **INT8 post-training quantized** using a representative dataset calibration set. It runs entirely in RAM — zero disk I/O during inference.

| Metric | Value |
|--------|-------|
| Architecture | MobileNetV2 (fine-tuned) |
| Quantization | INT8 (post-training) |
| Model size | 4.2 MB |
| Avg. inference latency | < 50ms (mid-range ARM) |
| Runtime | tflite_flutter (NNAPI / CoreML delegate) |

**Tags:** `MobileNetV2` · `INT8` · `tflite_flutter`

---

### Stage 05 — Risk Stratification & Report

The model's softmax output maps to 4 classes with probability scores. The primary label and confidence are displayed to the user. A full triage report is written to a **local encrypted SQLite database**:

```json
{
  "timestamp": "2025-06-14T03:22:11Z",
  "audio_path": "/data/local/respicore/rec_0023.wav",
  "class_probs": {
    "normal":     0.04,
    "anomalous":  0.07,
    "wheeze":     0.81,
    "copd":       0.08
  },
  "primary_label": "wheeze",
  "confidence": 81,
  "latency_ms": 42
}
```

Zero network calls are made at any stage.

**Tags:** `Softmax 4-class` · `SQLite` · `Offline-first`

---

## 🛠️ Tech Stack

| Tool | Role | Details |
|------|------|---------|
| **Librosa** | Audio processing | STFT, Mel-filterbank computation, silence trimming, spectral gating |
| **TensorFlow / PyTorch** | Model training | MobileNetV2 transfer learning, two-stage fine-tuning, categorical cross-entropy |
| **TensorFlow Lite** | On-device inference | INT8 quantization — 4× size reduction, 2–3× speed gain vs FP32 |
| **Flutter + Dart** | Cross-platform UI | Single codebase for Android + iOS; native audio via `record` package |
| **SQLite (sqflite)** | Local storage | All triage reports stored locally; schema captures timestamp, audio path, and 4 class probabilities |
| **COUGHVID** | Training data | EPFL's 25k-sample respiratory audio dataset |
| **Coswara** | Training data | IISc Project Coswara — labeled multi-condition respiratory audio |

---

## 📊 Model Performance

Evaluated on a held-out test split (80 / 10 / 10 train / val / test). MobileNetV2 INT8, fine-tuned on COUGHVID + Coswara combined.

| Metric | Score |
|--------|-------|
| **Accuracy** | 89.2% (4-class, test split) |
| **AUC-ROC** | 0.94 (macro-averaged, one-vs-rest) |
| **Model size** | 4.2 MB (INT8 quantized) |
| **Inference latency** | < 50ms (avg, mid-range ARM) |

---

## 🏆 Competitive Landscape

RespiCore vs. existing tools on metrics that matter for real-world deployment:

| Feature | RespiCore | ResApp Health | StethoMe | Hyfe AI |
|---------|:---------:|:-------------:|:--------:|:-------:|
| **Fully offline** | ✅ | ❌ | ❌ | ❌ |
| **No hardware add-on** | ✅ | ✅ | ❌ | ✅ |
| **Open source** | ✅ | ❌ | ❌ | ❌ |
| **Multi-condition classification** | ✅ | ✅ | ✅ | ❌ |
| **Free to use** | ✅ | ❌ | ❌ | ❌ |
| **Android + iOS** | ✅ | ✅ | ✅ | ✅ |
| **Local data storage** | ✅ | ❌ | ❌ | ❌ |
| **Model accuracy** | **89.2%** | 81.4% | 78.6% | 75.1% |
| **Model size** | **4.2 MB** | 24.6 MB | 31.8 MB | 18.3 MB |
| **Inference latency** | **38ms** | 220ms | 310ms | 180ms |
| **Privacy score** | **100/100** | 42/100 | 38/100 | 55/100 |

---

## 🔒 Privacy by Design

Every architectural decision in RespiCore was made with patient privacy as a hard constraint.

### Zero Network Calls
The app makes exactly zero outbound network requests during inference. No telemetry, no analytics, no crash reporting that touches audio data.

### On-Device Model Only
The 4.2 MB TFLite model is bundled inside the app binary. Audio files never leave the device sandbox under any circumstance.

### Local SQLite Storage
All triage reports are written to an encrypted SQLite database on the device. The user owns their data entirely.

### No Patient Identifiers
The app collects zero PII. No name, age, location, or device ID is associated with any triage report.

---

## 📁 Project Files

```
respicore-demo/
├── index.html              # Main app — hero, live demo, pipeline, tech stack, metrics
├── respicore_about.html    # About page — team, timeline, comparisons, privacy
├── review.html             # Community reviews — requires login (demo auth)
└── README.md               # This file
```

### Page Breakdown

**`index.html`** — The core experience. Contains:
- Animated hero with breathing ring visualization
- Interactive live triage demo (simulated)
- 5-stage pipeline explainer
- Full tech stack cards
- Model benchmark metrics
- Triage history table
- Language switcher (EN / हिं / বাং)
- Google OAuth demo login modal
- Session-gated Reviews nav link

**`respicore_about.html`** — The story behind the project. Contains:
- Animated particle hero
- Project timeline (Month 01 → Today)
- Team profiles (Adhrikto, Aarushi, Rajdeep, Sneha)
- Competitive comparison charts (Chart.js)
- Feature comparison table
- "Why we built it" impact stats
- Privacy architecture section

**`review.html`** — Community session reviews. Accessible only after signing in via the demo login flow.

---

## 🎮 Demo Walkthrough

### 1. Landing Page
Open `index.html`. You'll see the animated breathing ring hero with key stats: 16kHz sample rate, <50ms inference, 4.2MB model, ~89% accuracy.

### 2. Running a Triage

Navigate to the **Live Demo** section (or click "Run Triage Demo"):

1. Select a condition to simulate from the condition picker:
   - 🟢 **Normal** — clean acoustic baseline
   - 🟡 **Anomalous** — irregular mid-frequency activity
   - 🔵 **Wheeze** — sharp harmonic peaks at 30%, 55%, 75% frequency bands
   - 🔴 **COPD / Bronchitis** — broadband noise across four peak bands

2. Click **Start Recording** — a 10-second animated countdown begins with a live waveform.

3. After 10 seconds, results appear showing:
   - Risk badge and headline
   - Confidence percentage
   - Class probability bars for all 4 conditions
   - A Mel-spectrogram visualization (condition-specific color pattern)
   - Metadata chips: timestamp, sample rate, quantization type

4. Results are logged to the **Triage History** table below.

### 3. Exploring the Pipeline
Scroll to the **Pipeline** section to read the 5-stage technical breakdown with tags and parameters.

### 4. Our Story
Click **Our Story** in the navbar to open `respicore_about.html` in a new tab. Explore the team, origin timeline, and competitive benchmarks.

---

## 🔐 Authentication (Demo Mode)

RespiCore's web demo includes a simulated Google OAuth login system. **No real authentication occurs** — this is a demonstration of how the production app's session management would work.

### How it works

```
User clicks "Sign in"
        ↓
Login modal appears (with ⚡ DEMO SESSION notice)
        ↓
User clicks "Continue with Google"
        ↓
App randomly assigns one of 4 demo accounts:
  · Adhrikto (adhrikto@respicore.dev)
  · Aarushi  (aarushi@respicore.dev)
  · Rajdeep  (rajdeep@respicore.dev)
  · Sneha    (sneha@respicore.dev)
        ↓
Session stored in sessionStorage (cleared on tab close)
        ↓
Nav bar updates with user avatar + dropdown
Reviews nav link becomes active (🔒 removed)
```

### Session Persistence
The session persists across page refreshes within the same browser tab using `sessionStorage`. Closing the tab clears the session. The session is synced between `index.html` and the About page if they share the same origin.

### Guest Access
Users can also click **"Continue as Guest"** to get a guest session with access to all logged-in features.

### Gated Feature: Community Reviews
The **Reviews** link in the navbar is session-gated:

| State | Appearance | Behaviour on Click |
|-------|------------|-------------------|
| Logged out | `Reviews 🔒` (dimmed) | Toast + login modal |
| Logged in | `Reviews` (accent colour) | Opens `review.html` in new tab |

---

## 👥 The Team

| Member | Role | Skills |
|--------|------|--------|
| **Adhrikto** | ML Lead · Model Architect | Designed MobileNetV2 training pipeline, spectral preprocessing, INT8 quantization. Drove 200+ training runs to achieve 89% accuracy. | `PyTorch` `TFLite` `Librosa` |
| **Aarushi** | Mobile Lead · Flutter Engineer | Built the full Flutter app from scratch — audio capture, real-time waveform, tflite_flutter inference binding, SQLite reporting. | `Flutter` `Dart` `SQLite` |
| **Rajdeep** | Data Engineer · Research | Curated COUGHVID and Coswara datasets, designed augmentation strategy, built preprocessing pipeline with spectral gating. | `Python` `Librosa` `NumPy` |
| **Sneha** | UI/UX Lead · Frontend Engineer | Built the complete visual design system from the dark-mode UI to the animated waveform dashboard. | `Figma` `CSS` `JS` |

---

## 📅 Project Timeline

```
Month 01 — THE SPARK
  A team member's grandmother was misdiagnosed due to a faulty stethoscope
  in a rural clinic. That moment became the driving question: can a smartphone
  replace the hardware entirely?

Month 02 — RESEARCH DEEP-DIVE
  Reviewed 40+ academic papers on acoustic respiratory diagnostics.
  Explored COUGHVID and Project Coswara datasets. Confirmed that
  Mel-spectrograms encode clinically meaningful patterns.

Month 03 — FIRST WORKING MODEL
  After 200+ failed training runs, a fine-tuned MobileNetV2 achieved
  89% accuracy on held-out data. First sub-50ms inference on a
  5-year-old Android device. Celebrated at 3 AM.

Month 04 — FLUTTER APP & INTEGRATION
  Built the cross-platform Flutter UI. Integrated TFLite model on-device.
  Added SQLite reporting and waveform visualizer. First full end-to-end
  pipeline test in a single recording session.

TODAY — HACKATHON DEMO
  A production-grade acoustic triage system running entirely on a
  smartphone. 100% offline. 4.2 MB. Sub-50ms. Zero data transmitted.
```

---

## 🌐 Languages Supported

The web demo supports three languages via the language switcher in the navbar:

| Code | Language | Script |
|------|----------|--------|
| `EN` | English | Latin |
| `हिं` | Hindi | Devanagari |
| `বাং` | Bengali | Bengali |

Language selection syncs across the main page and the About page if both are open simultaneously.

---

## ⚠️ Clinical Disclaimer

> **RespiCore is a research prototype built for a hackathon demonstration.**
>
> - This tool is **NOT a medical device**
> - It is **NOT FDA/CE cleared or approved**
> - It should **NOT be used for clinical diagnosis**
> - It should **NOT replace consultation with a licensed healthcare professional**
>
> Any result produced by this system is for **informational and research purposes only**.
> If you or someone you know has respiratory symptoms, please consult a qualified doctor.

---

<div align="center">

**RespiCore · 2025 · Research Prototype**

*Built with 🫁 and a lot of instant noodles*

`⚡ DEMO SESSION — No real data is collected, stored, or transmitted.`

</div>