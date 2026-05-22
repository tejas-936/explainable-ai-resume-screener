import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import config


try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt')
    except Exception:
        pass

class ResumeHierarchicalDataset(Dataset):
    """
    A PyTorch Dataset for loading resumes from a CSV, segmenting them into
    sentences, and tokenizing each sentence hierarchically using a DistilBERT tokenizer.
    """
    def __init__(self, csv_file, tokenizer_name=None, max_sentences=None, max_words_per_sent=None):
        self.df = pd.read_csv(csv_file).dropna(subset=['Resume_str'])
        
        tokenizer_name = tokenizer_name or config.TOKENIZER_NAME
        self.max_sentences = max_sentences or config.MAX_SENTENCES
        self.max_words_per_sent = max_words_per_sent or config.MAX_WORDS_PER_SENT
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['Resume_str'])
        sentences = sent_tokenize(text)[:self.max_sentences]
        tokenized_sentences = []
        
        for sent in sentences:
            tokens = self.tokenizer(
                sent,
                truncation=True,
                max_length=self.max_words_per_sent,
                add_special_tokens=True,
                return_tensors="pt"
            )
            tokenized_sentences.append({
                'input_ids': tokens['input_ids'].squeeze(0),
                'attention_mask': tokens['attention_mask'].squeeze(0)
            })
            
        return tokenized_sentences

def hierarchical_collate_fn(batch):
    """
    Collate function to pad hierarchical document-sentence batches to uniform length.
    Returns padded input IDs, attention masks, and a sentence mask (indicating valid sentences).
    """
    max_sents_in_batch = max(len(doc) for doc in batch)
    max_words_in_batch = max(max((len(sent['input_ids']) for sent in doc), default=0) for doc in batch)
    batch_size = len(batch)
    
    padded_input_ids = torch.zeros((batch_size, max_sents_in_batch, max_words_in_batch), dtype=torch.long)
    padded_attention_masks = torch.zeros((batch_size, max_sents_in_batch, max_words_in_batch), dtype=torch.long)
    sentence_mask = torch.zeros((batch_size, max_sents_in_batch), dtype=torch.long)
    
    for doc_idx, doc in enumerate(batch):
        for sent_idx, sent in enumerate(doc):
            word_len = len(sent['input_ids'])
            padded_input_ids[doc_idx, sent_idx, :word_len] = sent['input_ids']
            padded_attention_masks[doc_idx, sent_idx, :word_len] = sent['attention_mask']
            sentence_mask[doc_idx, sent_idx] = 1
            
    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_masks,
        'sentence_mask': sentence_mask
    }

def tokenize_single_resume(text, tokenizer, max_sentences=None, max_words_per_sent=None):
    """
    Utility function to convert a raw string of a single resume into padded
    hierarchical tensors matching the model inputs (batch size 1).
    """
    max_sentences = max_sentences or config.MAX_SENTENCES
    max_words_per_sent = max_words_per_sent or config.MAX_WORDS_PER_SENT
    
    sentences = sent_tokenize(text)[:max_sentences]
    tokenized_sentences = []
    
    for sent in sentences:
        tokens = tokenizer(
            sent,
            truncation=True,
            max_length=max_words_per_sent,
            add_special_tokens=True,
            return_tensors="pt"
        )
        tokenized_sentences.append({
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0)
        })
        

    batch = hierarchical_collate_fn([tokenized_sentences])
    return batch, sentences
