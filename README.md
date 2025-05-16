# Using Transfer Learning to Detect Road-Surface Defects:  
*An End-to-End Colab Workflow for Data Collection, Model Training, and Live Inference*

This lab invites you to explore how modern computer-vision techniques—specifically transfer learning on pretrained CNNs—can be harnessed to see road-surface defects at scale. Borrowing real-world imagery scraped directly from the web, the workflow guides you through (1) automating data collection for any target object (e.g., potholes, crosswalks), (2) adapting a lightweight MobileNetV2 backbone to recognize that object with only a few lines of code, and (3) deploying the resulting model to score brand-new photos in real time. By the end, you’ll link pixels to pavement condition, raising a broader question: What happens when machines start auditing our streets, one image at a time?

## Why This Lab?

Computer-vision models are only as good as the data pipeline behind them.  
Across **three concise notebooks** you will:

1. **Collect & clean** a balanced image dataset with a few lines of code.  
2. **Train & validate** a lightweight MobileNetV2 classifier using transfer learning.  
3. **Deploy & test** the model on your own photos—seeing both **positive** *and* **negative** probabilities.

Everything runs in Google Colab, so the only requirement is a Google Drive account.


## Quick Launch

| Notebook | Link|
|----------|---------------|
| **Module 1 — Data Collection & Preprocessing** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](<https://colab.research.google.com/drive/1gEEDpoqwdP4CKok20QzV66b-BKMe2PYP?usp=drive_link>)  |
| **Module 2 — Transfer Learning & Cross-Validation** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](<https://colab.research.google.com/drive/1a1m8uVH7lv_vxU1MwrhhLzcbCJEXlJhG?usp=drive_link>) |
| **Module 3 — Inference & Prediction** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](<https://colab.research.google.com/drive/1SPBEpMy1Nnu2YN21h4vMeKPLVg0sgZDI?usp=drive_link>)  |



## Lab Roadmap

### Module 1 · Data Collection & Preprocessing  
*Scrape images with DuckDuckGo, balance positive/negative classes, and package them into a `tf.data` pipeline.*

1. **Cell A** &nbsp;Mount Google Drive.  
2. Set the `TARGET` variable (e.g. `"potholes"`, `"zebra_crossing"`).  
3. Run the notebook to download, clean, resize, and save images:
```plaintext
data/
└── raw_images/
├── road_<TARGET>/
└── road_without_<TARGET>/
```
4. The helper `create_dataset()` stored in `ml_utils` will be reused in Module 2.

### Module 2 · Transfer Learning, Cross-Validation & Evaluation  
*Fine-tune MobileNetV2 and judge performance with stratified 5-fold splits.*

- Mount Drive and import `ml_utils`.  
- Build a MobileNetV2 backbone + small classification head.  
- `StratifiedShuffleSplit` keeps class ratios identical across folds.  
- Auto-generated plots help you spot **under-fitting** (both curves low) or **over-fitting** (training ↑, validation ↔/↓).

### Module 3 · Inference & Deployment  
*Load the saved `.h5` model and predict on new images straight from your laptop.*

| Cell | What it does |
|------|--------------|
| **Cell A – Positive** | Upload an image → prints *Positive-class* confidence (`TARGET` present) |
| **Cell B – Negative** | Upload an image → prints *Negative-class* confidence (`TARGET` absent) |


## Repository Structure
```plaintext
<repo-root>/
├── Module1.ipynb
├── Module2.ipynb
├── Module3.ipynb
├── ml_utils.py # shared helper functions
├── data/ # created automatically in Module 1
│ └── raw_images/ …
└── trained_models/ # .h5 saved in Module 2
```

---

## Guiding Questions

1. How does enforcing a **1 : 1 class balance** affect accuracy?  
2. What do your training curves reveal about **over-fitting** vs. **under-fitting**?  
3. Does the model ever assign high confidence to the wrong class? Why might that happen?  

Reflect on these after completing Module 3.

---

## Deliverables


---
