# Persian Emotion Classification with BERT
🎭 **A Persian emotion classification pipeline** using a fine-tuned BERT model (`HooshvareLab/bert-base-parsbert-uncased`), from raw tweet data to deployment-ready inference.


## 📝 Short Description

This repository implements a complete workflow for detecting emotions in Persian tweets. It merges multiple datasets, cleans and preprocesses text, fine-tunes a BERT-based model with weighted loss, and provides tools for evaluation and inference.

---

## 🌟 Features & Highlights

* **Multi-Dataset Integration**: Combines ArmanEmo, EmoPars, and Short Persian Emo.
* **Text Cleaning**: Parsivar normalization, character mapping, URL/emoji/punctuation removal.
* **Imbalance Handling**: Class weights with optional resampling steps.
* **Emotion Categories**: Detects 6 emotions — `ANGRY`, `FEAR`, `HAPPY`, `HATE`, `SAD`, `SURPRISE`.
* **Custom Training**: Lightweight wrapper for weighted-loss fine-tuning with Hugging Face Trainer.
* **Comprehensive Metrics**: Tracks accuracy, precision, recall, F1, and confusion matrix.
* **Deployment-Ready**: Save/load model and tokenizer; Hugging Face Hub integration.



---

## ⚙️ Usage

Users can load the fine-tuned model and tokenizer directly from the Hugging Face Hub:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "aref-j/emotion-classifier-bert-fa-v1"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create the classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Example usage
result = classifier("چه هوای زیبایی امروز است")
print(result)  # e.g. HAPPY
```

---

## 🗂️ Data Preparation

1. **Loading & Formatting**:

   * Standardize raw datasets to `text`, `label`, `label_name`.
   * Convert multi-label EmoPars to single-label via dominant-label rule.
2. **Cleaning**:

   * Remove duplicates, nulls.
   * Normalize Persian text (Parsivar, character mapping, remove diacritics).
   * Strip URLs, mentions, hashtags, emojis, punctuation, digits, extra spaces.
3. **Merging & Splitting**:

   * Merge three datasets; dedupe.
   * Split into 90% train / 10% validation; hold out ArmanEmo for testing.
4. **Saving**:

   * Save Pandas CSVs and Hugging Face `DatasetDict` for easy reuse.

---

## 🤖 Model

* **Architecture**: BERT base (`bert-base-parsbert-uncased`) with a sequence classification head.
* **Training**:

  * Batch size: 32
  * Epochs: 6
  * Learning rate: 1e-5
  * Weighted cross-entropy loss to address imbalance.
  * Evaluation and checkpointing at each epoch.
  * Early stopping: Stops training if validation loss doesn’t improve after 2 consecutive epochs

---

## 📈 Evaluation

The model was evaluated on the held-out **ArmanEmo** test set. Key performance on this dataset:

* **Test Accuracy:** 70.88%
* **Macro F1-Score:** 66.35%

This demonstrates robust generalization to real-world Persian tweet data.

---

## 📁 Directory Structure

Due to licensing restrictions on the source datasets (ArmanEmo, EmoPars, and ShortPersianEmo), 
the raw and processed data files are **not included** in this repository. You must download 
the source datasets yourself and run the preprocessing notebook to regenerate them locally.

### Expected folder structure (generated locally, not tracked in git)

```
Persian-Emotion-Classification-BERT/
├── data/                              
│   ├── raw/                          # Original emotion datasets
│   │   ├── ArmanEmo/
│   │   ├── EmoPars/
│   │   └── ShortPersianEmo/
│   ├── processed/                    # Cleaned and formatted data
│   │   ├── CSV/
│   │   └── DatasetDict/             # HuggingFace format
│
├── notebooks/
│   ├── 01_preprocessing.ipynb       # Preprocessing and dataset formatting
│   └── 02_tuning.ipynb              # BERT fine-tuning code
│
├── README.md                        # Project documentation
             

```


### Sources
- **ArmanEmo**: [github.com/Arman-Rayan-Sharif/arman-text-emotion](https://github.com/Arman-Rayan-Sharif/arman-text-emotion) (CC BY-NC-SA 4.0)
- **EmoPars**: [github.com/nazaninsbr/Persian-Emotion-Detection](https://github.com/nazaninsbr/Persian-Emotion-Detection)
- **ShortPersianEmo**: [github.com/vkiani/ShortPersianEmo](https://github.com/vkiani/ShortPersianEmo) (GPL-3.0)

### How to reproduce
1. Download each dataset from the links above and place it under `data/raw/<DatasetName>/`
2. Run `notebooks/01_preprocessing.ipynb` to clean, harmonize, and generate `data/processed/`
3. Run `notebooks/02_tuning.ipynb` to fine-tune the model

---

## 📄 License

The **code** in this repository is released under the MIT License.

> **Note:** This license applies only to the code, not to the datasets. The source datasets (ArmanEmo, EmoPars, ShortPersianEmo) each carry their own separate licenses — see the **Sources** section above. Please review and comply with each dataset's license independently before use.

---

## 🤝 Acknowledgments

* **Arman-Rayan-Sharif** for the ArmanEmo dataset.
* **Nazanin SBR** for the EmoPars dataset.
* **Vahid Kiani** for the Short Persian Emo dataset.
* **HooshvareLab** for their `bert-base-parsbert-uncased` model and ParsBERT research.
* **Parsivar** library for Persian text normalization.
* Hugging Face Transformers team for the training framework.
