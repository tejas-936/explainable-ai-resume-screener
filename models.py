import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import DistilBertModel
import config

class PositionalEncoding(nn.Module):
    """
    Applies standard positional encoding to input word embeddings.
    """
    def __init__(self, d_model, max_len=3000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class HierarchicalEncoder(nn.Module):
    """
    Hierarchical document encoder.
    First encodes words using DistilBERT (with frozen embeddings).
    Then pools words into sentences (using CLS token) and passes sentences through a Transformer Encoder.
    """
    def __init__(self, word_encoder_name=None, sent_hidden_dim=None, num_sent_layers=None):
        super().__init__()
        
        word_encoder_name = word_encoder_name or config.WORD_ENCODER_NAME
        sent_hidden_dim = sent_hidden_dim or config.SENT_HIDDEN_DIM
        num_sent_layers = num_sent_layers or config.NUM_SENT_LAYERS
        
        self.word_encoder = DistilBertModel.from_pretrained(word_encoder_name)
        

        for param in self.word_encoder.embeddings.parameters():
            param.requires_grad = False
            
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.word_encoder.config.dim, 
            nhead=8, 
            dim_feedforward=2048, 
            batch_first=True
        )
        self.sent_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_sent_layers)
        self.output_dim = sent_hidden_dim

    def forward(self, input_ids, attention_mask, sentence_mask):
        B, S, W = input_ids.size()
        flat_input_ids = input_ids.view(B * S, W)
        flat_attn_mask = attention_mask.view(B * S, W)
        

        word_outputs = self.word_encoder(flat_input_ids, attention_mask=flat_attn_mask)
        cls_embeddings = word_outputs.last_hidden_state[:, 0, : ]
        sent_sequence = cls_embeddings.view(B, S, -1)
        

        src_key_padding_mask = (sentence_mask == 0)
        doc_embeddings = self.sent_encoder(sent_sequence, src_key_padding_mask=src_key_padding_mask)
        

        mask_expanded = sentence_mask.unsqueeze(-1).expand(doc_embeddings.size()).float()
        sum_embeddings = torch.sum(doc_embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        global_doc_embedding = sum_embeddings / sum_mask
        
        return global_doc_embedding

class HTVAE(nn.Module):
    """
    Hierarchical Transformer Variational Autoencoder (HTVAE).
    Encodes hierarchical text documents into a latent distribution (mu, logvar),
    then reconstructs the document using a Transformer Decoder.
    """
    def __init__(self, latent_dim=None, vocab_size=None, hidden_dim=None):
        super().__init__()
        
        latent_dim = latent_dim or config.LATENT_DIM
        vocab_size = vocab_size or config.VOCAB_SIZE
        hidden_dim = hidden_dim or config.HIDDEN_DIM
        
        self.encoder = HierarchicalEncoder(sent_hidden_dim=hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        
        self.decoder_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=3000)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.vocab_projection = nn.Linear(hidden_dim, vocab_size)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, input_ids, attention_mask, sentence_mask, decoder_input_ids):

        global_doc_emb = self.encoder(input_ids, attention_mask, sentence_mask)
        

        mu = self.fc_mu(global_doc_emb)
        logvar = self.fc_logvar(global_doc_emb)
        z = self.reparameterize(mu, logvar)
        

        memory = self.latent_to_hidden(z).unsqueeze(1)
        

        decoder_emb = self.decoder_emb(decoder_input_ids)
        decoder_emb = self.pos_encoder(decoder_emb)
        
        seq_len = decoder_input_ids.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_ids.device)
        
        decoder_output = self.decoder(tgt=decoder_emb, memory=memory, tgt_mask=tgt_mask)
        logits = self.vocab_projection(decoder_output)
        
        return logits, mu, logvar, z
