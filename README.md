# Visual Object Tracking (VOT) Portfolio: A Progressive Framework

This repository contains a comprehensive portfolio of 5 Computer Vision projects. The entire codebase is structured as a step-by-step methodological journey, tracing the historical and technological evolution of Visual Object Tracking—from pixel-level intensity variations to modern long-term deep learning systems.

## 📈 The Tracking Evolution: A Step-by-Step Overview

### 📦 Phase 1: From Pixels to Patches (Optical Flow & Lucas-Kanade)
- **Concept:** Moving from independent pixel tracking to structural patches.
- **Implementation:** Began with baseline pixel-level **Optical Flow** intensity tracking, advancing to the **Lucas-Kanade Tracker**. This transition introduces local spatial consistency assumptions to track image patches rather than isolated points.

### 📦 Phase 2: Overcoming Geometric Constraints (Mean-Shift)
- **Concept:** Transitioning from rigid geometric templates to statistical appearance profiles.
- **Implementation:** Implemented the **Mean-Shift Tracker**. By modeling the target using color histograms (RGB/HSV) and optimizing via Bhattacharyya distance, this approach breaks the rigid spatial constraints of standard spatial tracking, offering resilience against non-rigid deformations.

### 📦 Phase 3: Tracking as Local Classification (Correlation Filters)
- **Concept:** Utilizing pattern recognition and fast local classification.
- **Implementation:** Developed **Discriminative Correlation Filter** tracking. Starting from the baseline **MOSSE** filter and moving to multi-channel configurations with **HOG (Histogram of Oriented Gradients)** features, this phase reframes tracking as an optimized local classification/template-matching task in the frequency domain.

### 📦 Phase 4: Predictability & Dynamics (Recursive Bayesian Filters)
- **Concept:** Integrating observable data with predictive dynamic models to handle severe occlusions.
- **Implementation:** Explored **Recursive Bayesian Estimation** to decouple tracking into a dual-component system: an observation model and a predictive state model. Implemented linear **Kalman Filtering** and complex non-linear **Particle Filters**, drastically increasing robustness by anticipating target trajectories even when temporarily out of sight.

### 📦 Phase 5: Deep Learning & Long-Term Autonomy (SiamFC & Failure Detection)
- **Concept:** Boosting feature representations with Deep Learning and handling structural target disappearance.
- **Implementation:** Combined state-of-the-art methodologies by utilizing a **Siamese Fully-Convolutional Network (SiamFC)** in PyTorch for robust deep feature extraction. Finalized the pipeline with a **Long-Term Tracking** state machine that embeds explicit tracking failure detection and an automated search/detection mechanism to recover the target after prolonged full occlusions.

## 🛠️ Tech Stack & Core Libraries
- **Language:** Python
- **Deep Learning Framework:** PyTorch
- **Computer Vision Pipeline:** OpenCV, SciPy, NumPy, Matplotlib