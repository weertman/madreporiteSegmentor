# madreporiteSegmentor


## Prerequisites

- Python 3.11 or newer
- Conda package manager

## Setup Instructions

1. Create a virtual environment:
```bash
conda create --name projectName python=3.11
conda activate projectName
```

2. Install PyTorch and related packages:
   - Visit [PyTorch's official installation page](https://pytorch.org/get-started/locally/) to get the command specific to your system
   - For MacOS users:
   ```bash
   pip3 install torch torchvision torchaudio
   ```
   
3. Install other required packages:
```bash
pip install pathlib
pip install matplotlib
pip install opencv-python
pip install albumentations
pip install transformers
pip install pandas
pip install wandb
pip install jupyter
pip install "numpy<2"
```

4. Login to Weights & Biases:
```bash
wandb login <your-api-key>
```
