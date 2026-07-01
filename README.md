# CountEx: Fine-Grained Counting via Exemplars and Exclusion
[Demo](https://huggingface.co/spaces/yifehuang97/CountEx)
[Data](https://huggingface.co/collections/BBVisual/cocount)
[Model](https://huggingface.co/collections/BBVisual/countex)

> **Note:** All training and inference use `bf16`, so only NVIDIA Ampere architecture (or newer) GPUs are supported.  
> We conduct all experiments on NVIDIA RTX A5000 GPUs.

## 📌 Important Note on Corrected Test Annotations

We identified a small annotation issue in the test-set evaluation labels and corrected the affected annotations. The issue is limited to the evaluation labels and does not affect the training annotations used by CountEx, since CountEx is trained with dot annotations.

The released dataset now contains the corrected test labels. Because the paper was submitted before this correction, the results reported in the paper were computed using the original test labels. The table below reports CountEx results on the corrected labels. The differences from the paper results are small, and the overall conclusions remain consistent.

| Split | # Corrected Images | MAE (Paper Reported ) | MAE (Corrected) | Abs Δ MAE | RMSE (Paper Reported) | RMSE (Corrected) | Abs Δ RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Food | 30 | 37.04 | 37.40 | 0.36 | 50.58 | 51.30 | 0.72 |
| Home | 0 | 24.16 | 24.16 | 0.00 | 34.87 | 34.87 | 0.00 |
| Desk | 18 | 31.18 | 27.89 | 3.29 | 51.90 | 46.47 | 5.43 |
| Misc | 72 | 23.82 | 22.97 | 0.85 | 32.68 | 31.88 | 0.80 |
| Game | 30 | 16.84 | 16.84 | 0.00 | 24.26 | 24.26 | 0.00 |
| KC (Overall) | 150 | 12.72 | 11.20 | 1.52 | 23.99 | 20.32 | 3.67 |

For the Game split, although 30 images/frames were updated, the corrected counts are very close to the original counts, so the final MAE/RMSE remain unchanged after rounding.

## Setup

1. **Clone the repository**
```bash
   git clone <repository-url>
   cd CountEx
```

2. **Create and activate a conda environment**
```bash
   conda create -n countex python=3.10.18
   conda activate countex
```

3. **Install dependencies**

- **Install PyTorch with CUDA 12.1**
```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```
   - **Install other packages**
```bash
pip install transformers==4.42.0
pip install deepspeed==0.17.0
pip install accelerate==1.6.0
pip install wandb
pip install datasets
pip install matplotlib
pip install scipy
```
   - **(Training, for gcc-11)**
```bash
conda install -c conda-forge gcc=11 gxx=11
```

You can also run all the above steps by executing the provided script:
```bash
bash src/eval_env_setup.sh
```

## How to Run Evaluation / Training

CountEx supports two evaluation settings on the CoCount dataset:

- **KC-setting (Known-Category)**: All five supercategories are available during training. This setting evaluates the model's performance on categories it has seen during training.
- **NC-setting (Novel-Category)**: Each supercategory is held out as the test set while training on the remaining four. This setting evaluates zero-shot generalization to novel object categories.

### Prerequisites

1. **Set your Hugging Face token**  
   Edit the `export HF_TOKEN=...` line in the evaluation scripts located in `src/scripts/eval/`, or export it manually in your terminal:
```bash
   export HF_TOKEN=your_huggingface_token_here
```

### Running Evaluation

**For KC-setting:**
```bash
cd src
bash scripts/eval/KC.sh
```

**For NC-setting** (choose the supercategory to hold out):
```bash
cd src
# Test on Food (train on Home, Desk, Misc, Game)
bash scripts/eval/nc_food.sh

# Test on Home (train on Food, Desk, Misc, Game)
bash scripts/eval/nc_home.sh

# Test on Desk (train on Food, Home, Misc, Game)
bash scripts/eval/nc_desk.sh

# Test on Misc (train on Food, Home, Desk, Game)
bash scripts/eval/nc_misc.sh

# Test on Game (train on Food, Home, Desk, Misc)
bash scripts/eval/nc_game.sh
```

## Training

### For KC-setting
```bash
bash scripts/train/kc.sh
```

We provide wandb training logs with this codebase for the KC-setting as a reference: [📊 View training logs](https://wandb.ai/yife/CountEx_KC/)

### For NC-setting (choose the supercategory to hold out):
```bash
# Train on Home, Desk, Misc, Game; test on Food
bash scripts/train/nc_food.sh

# Train on Food, Desk, Misc, Game; test on Home
bash scripts/train/nc_home.sh

# Train on Food, Home, Misc, Game; test on Desk
bash scripts/train/nc_desk.sh

# Train on Food, Home, Desk, Game; test on Misc
bash scripts/train/nc_misc.sh

# Train on Food, Home, Desk, Misc; test on Game
bash scripts/train/nc_game.sh
```

## Live Demo

You can try CountEx directly in your browser via our Hugging Face Spaces demo:

👉 [CountEx Interactive Demo](https://huggingface.co/spaces/yifehuang97/CountEx)
