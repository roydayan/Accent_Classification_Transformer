# Accent_Classification_Transformer
Using a pretrained speech transformer for accent classification.

## Project Overview

This project fine-tunes **Wav2Vec2** for **accent classification**, using **data augmentation** and **LoRA (Low-Rank Adaptation)** to improve efficiency and generalization.

---

## Project Structure

- `DataGenerating.ipynb` → Prepares and organizes the dataset in Google Drive.
- `OGDATAtraining.ipynb` → Trains Wav2Vec2 on the **original dataset** (without augmentation).
- `AugmentedData.ipynb` → Trains Wav2Vec2 using **augmented** speech samples.
- `Lora.ipynb` → Fine-tunes Wav2Vec2 using **LoRA** for efficient adaptation.

---

## Requirements

### **Install Dependencies**
Before running any notebooks, install the necessary Python libraries:
```bash
pip install transformers datasets torch librosa soundfile

### **Set Up Google Drive**
All project-related files will be stored inside **Google Drive** under the `/My Drive/Deep_Project/` directory.  

### **Instructions for Setup**
1. **Before running any notebooks**, create a folder named **`Deep_Project/`** in **Google Drive**.  

2. Inside `Deep_Project/`, **create the following subdirectories**:
   - `Data/` 
   - `new_data/`  
   - `AugmentedData/`  
   - `TestData/`  
   - `saved_models/`
      -`augmented_model/`

  Each subdirectory serves a specific purpose:
  - **`new_data/`** → Stores the processed dataset, with each speaker having their own folder (`speaker1/`, `speaker2/`, ...).  
  - **`AugmentedData/`** → Stores augmented versions of the dataset, also organized by speaker.  
  - **`TestData/`** → Stores the dataset used for evaluating model performance.  
  - **`saved_models/`** → Stores trained model checkpoints, including `augmented_model/` where the fine-tuned model is saved.  
  - **`wav2vec2-lora-adapt/`, `wav2vec2-lora/`, `wav2vec2-base-fine-tuned/`** → Optional folders for different versions of fine-tuned models.  

3. The notebooks will **automatically generate and save files** inside these directories.  

4. If running in **Google Colab**, make sure to mount Google Drive before starting:  

   ```python
   from google.colab import drive
   drive.mount('/content/drive')

5. The files in /My Drive/ should be shortcuts pointing to their actual locations in /My Drive/Deep_Project/
  /My Drive/
    ├── Data/             # Original Arctic dataset
    ├── new_data/         # Processed dataset (organized by speakers)
    ├── AugmentedData/    # Augmented dataset (if using augmentation)
    ├── TestData/        # Test (Archive) dataset

6. Download & Extract Datasets. The datasets are provided as ZIP archives and should be extracted inside Google Drive. Arctic dataset into /MyDrive/Data, Archive into /MyDrive/TestData


### **How to Run**
1. Prepare Data → Run DataGenerating.ipynb to structure and preprocess the dataset.

2. Train on Original Data → Run OGDATAtraining.ipynb to fine-tune Wav2Vec2 without augmentation.

3. Train on Augmented Data (Optional) → Run AugmentedData.ipynb to train the model on data-augmented speech samples.

4. Fine-Tune with LoRA (Optional) → Run Lora.ipynb for efficient fine-tuning using LoRA to reduce memory usage while training.

5️. Evaluate Performance → Compare model results on different datasets to assess generalization and accent classification accuracy.


