import torch

# 설정값
SAMPLE_RATE = 44100
NUM_MELS = 64
BATCH_SIZE = 32
NUM_CLASSES = 15
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
PATIENCE = 10  # Early stopping patience
MAX_FRAMES = 128
CHECKPOINT_DIR = "./checkpoints"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE='cpu'