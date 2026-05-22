import os


TOKENIZER_NAME = 'distilbert-base-uncased'
WORD_ENCODER_NAME = 'distilbert-base-uncased'
MAX_SENTENCES = 30
MAX_WORDS_PER_SENT = 40
SENT_HIDDEN_DIM = 768
NUM_SENT_LAYERS = 2
LATENT_DIM = 128
VOCAB_SIZE = 30522
HIDDEN_DIM = 768


EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
ACCUMULATION_STEPS = 4
BETA = 0.1


RAW_CSV_PATH = os.environ.get("RAW_CSV_PATH", "Resume.csv")
CLEANED_CSV_PATH = "Full_Resume.csv"
CHECKPOINT_PATH = "htvae_production_checkpoint.pth"


BIAS_SWAP_OPTIONS = [
    "None", 
    "Swap to Female Pronouns/Names", 
    "Swap to Male Pronouns/Names", 
    "Swap to Minority Demographics"
]
BIAS_L2_THRESHOLD = 0.3
