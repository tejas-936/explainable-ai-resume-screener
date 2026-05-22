import argparse
import os
import pandas as pd
import torch
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader

import config
from dataset import ResumeHierarchicalDataset, hierarchical_collate_fn, tokenize_single_resume
from models import HTVAE
from train import train_htvae
from explainability import ExplainableRecruitmentSuite, generate_counterfactual

def run_preprocess(args):
    """
    Cleans raw resume CSV dataset by removing missing data and saves it.
    """
    raw_path = args.raw_csv or config.RAW_CSV_PATH
    cleaned_path = args.cleaned_csv or config.CLEANED_CSV_PATH

    print(f"Loading raw dataset from: {raw_path}")
    if not os.path.exists(raw_path):
        print(f"Error: Raw dataset file '{raw_path}' not found.")
        print("Please place the Resume.csv in the workspace or specify its path using --raw-csv")
        return

    df = pd.read_csv(raw_path)
    if 'Resume_str' not in df.columns:
        print("Error: Column 'Resume_str' not found in raw CSV.")
        return

    df_clean = df.dropna(subset=['Resume_str'])
    df_clean.to_csv(cleaned_path, index=False)
    print(f"Dataset preprocessed. Cleaned dataset saved to: {cleaned_path}")
    print(f"Total valid resumes: {len(df_clean)}")

def run_train(args):
    """
    Trains the HTVAE model on the preprocessed CSV dataset.
    """
    cleaned_path = args.cleaned_csv or config.CLEANED_CSV_PATH
    checkpoint_path = args.checkpoint or config.CHECKPOINT_PATH

    print(f"Checking for preprocessed dataset: {cleaned_path}")
    if not os.path.exists(cleaned_path):
        print(f"Preprocessed file not found. Preprocessing raw dataset '{config.RAW_CSV_PATH}' first...")

        class PreprocessArgs:
            raw_csv = config.RAW_CSV_PATH
            cleaned_csv = cleaned_path
        run_preprocess(PreprocessArgs())
        
        if not os.path.exists(cleaned_path):
            print("Aborting training: preprocessed data could not be generated.")
            return

    print("Initializing tokenizer and dataset...")
    dataset = ResumeHierarchicalDataset(
        csv_file=cleaned_path,
        tokenizer_name=config.TOKENIZER_NAME,
        max_sentences=config.MAX_SENTENCES,
        max_words_per_sent=config.MAX_WORDS_PER_SENT
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=hierarchical_collate_fn,
        drop_last=True
    )

    print("Initializing HTVAE Model...")
    model = HTVAE(
        latent_dim=config.LATENT_DIM,
        vocab_size=config.VOCAB_SIZE,
        hidden_dim=config.HIDDEN_DIM
    )

    print("Starting training execution...")
    train_htvae(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        lr=args.lr,
        accumulation_steps=args.accumulation_steps,
        checkpoint_path=checkpoint_path
    )

def run_audit(args):
    """
    Runs diagnostic checks (attributions & counterfactual checks) on a single resume.
    """
    checkpoint_path = args.checkpoint or config.CHECKPOINT_PATH
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint file '{checkpoint_path}' not found.")
        print("Please train the model first using: python main.py train")
        return


    resume_text = ""
    if args.resume_file:
        if not os.path.exists(args.resume_file):
            print(f"Error: Resume file '{args.resume_file}' not found.")
            return
        with open(args.resume_file, 'r', encoding='utf-8') as f:
            resume_text = f.read()
    elif args.resume_text:
        resume_text = args.resume_text
    else:

        cleaned_path = config.CLEANED_CSV_PATH
        if not os.path.exists(cleaned_path):
            print("No input resume text provided and cleaned dataset not found.")
            print("Please specify --resume-text or run preprocessing first.")
            return
        print(f"No resume input provided. Loading first resume from: {cleaned_path}")
        df = pd.read_csv(cleaned_path)
        resume_text = str(df.iloc[0]['Resume_str'])

    print("\n--- RESUME PREVIEW (FIRST 300 CHARACTERS) ---")
    print(resume_text[:300] + "...")
    print("-------------------------------------------\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(config.TOKENIZER_NAME)
    
    print("Loading HTVAE model...")
    model = HTVAE(
        latent_dim=config.LATENT_DIM,
        vocab_size=config.VOCAB_SIZE,
        hidden_dim=config.HIDDEN_DIM
    )
    
    print(f"Loading model state from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    suite = ExplainableRecruitmentSuite(model, tokenizer)


    print("Tokenizing input resume...")
    inputs, sentences = tokenize_single_resume(
        resume_text, 
        tokenizer,
        max_sentences=config.MAX_SENTENCES,
        max_words_per_sent=config.MAX_WORDS_PER_SENT
    )
    

    inputs_dev = {k: v.to(device) for k, v in inputs.items()}


    print("Calculating sentence-level attribution (occlusion impact)...")
    attributions = suite.sentence_level_attribution(
        inputs_dev['input_ids'],
        inputs_dev['attention_mask'],
        inputs_dev['sentence_mask']
    )

    print("\n=== SENTENCE ATTRIBUTION RESULTS (TOP 5) ===")
    for item in attributions[:5]:
        sent_idx = item['sentence_idx']
        score = item['importance']
        print(f"S{sent_idx+1} [Importance Score: {score:.4f}]: \"{sentences[sent_idx]}\"")


    print("\nRunning counterfactual gender/demographic audit...")
    cf_type = args.bias_swap
    if cf_type == "none":
        print("Bias check skipped (swap type is set to 'none').")
        return
        
    swap_name = {
        "female": "Swap to Female Pronouns/Names",
        "male": "Swap to Male Pronouns/Names",
        "minority": "Swap to Minority Demographics"
    }[cf_type]

    cf_text = generate_counterfactual(resume_text, swap_name)
    cf_inputs, _ = tokenize_single_resume(
        cf_text,
        tokenizer,
        max_sentences=config.MAX_SENTENCES,
        max_words_per_sent=config.MAX_WORDS_PER_SENT
    )
    cf_inputs_dev = {k: v.to(device) for k, v in cf_inputs.items()}

    audit_res = suite.counterfactual_audit(
        (inputs_dev['input_ids'], inputs_dev['attention_mask'], inputs_dev['sentence_mask']),
        (cf_inputs_dev['input_ids'], cf_inputs_dev['attention_mask'], cf_inputs_dev['sentence_mask']),
        threshold=config.BIAS_L2_THRESHOLD
    )

    print("\n=== COUNTERFACTUAL AUDIT SUMMARY ===")
    print(f"Swap Option:         {swap_name}")
    print(f"Cosine Similarity:   {audit_res['cosine_similarity']:.4f}")
    print(f"L2 Latent Shift:     {audit_res['l2_distance']:.4f}")
    status = "FAILED (Bias Detected)" if audit_res['bias_detected'] else "PASSED (No Demographic Bias)"
    print(f"Fairness Evaluation: {status}")
    print("====================================")

def main():
    parser = argparse.ArgumentParser(description="XAI Recruitment Suite CLI Runner")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")


    parser_prep = subparsers.add_parser("preprocess", help="Preprocess raw resume dataset")
    parser_prep.add_argument("--raw-csv", type=str, help="Path to raw CSV file")
    parser_prep.add_argument("--cleaned-csv", type=str, help="Path to write preprocessed CSV file")


    parser_train = subparsers.add_parser("train", help="Train the HTVAE Model")
    parser_train.add_argument("--cleaned-csv", type=str, help="Path to preprocessed CSV file")
    parser_train.add_argument("--checkpoint", type=str, help="Path to save model checkpoint")
    parser_train.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of epochs to train")
    parser_train.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser_train.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser_train.add_argument("--accumulation-steps", type=int, default=config.ACCUMULATION_STEPS, help="Gradient accumulation steps")


    parser_audit = subparsers.add_parser("audit", help="Run explainability audit on a resume")
    parser_audit.add_argument("--resume-text", type=str, help="Raw text of the resume to analyze")
    parser_audit.add_argument("--resume-file", type=str, help="Path to file containing resume text")
    parser_audit.add_argument("--checkpoint", type=str, help="Path to loaded model checkpoint")
    parser_audit.add_argument("--bias-swap", type=str, choices=["none", "female", "male", "minority"], default="female", help="Demographic swap target to evaluate bias")

    args = parser.parse_args()

    if args.command == "preprocess":
        run_preprocess(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "audit":
        run_audit(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
