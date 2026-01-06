# CountEx
[Demo](https://huggingface.co/spaces/yifehuang97/CountEx)
[Data](https://huggingface.co/collections/BBVisual/cocount)
[Model](https://huggingface.co/collections/BBVisual/countex)

> **Note:** All training and inference use `bf16`, so only NVIDIA Ampere architecture (or newer) GPUs are supported.  
> We conduct all experiments on NVIDIA RTX A5000 GPUs.

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
