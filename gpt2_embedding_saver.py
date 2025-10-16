"""
gpt2_embeddings_pt_saver.py

Save GPT-2 embeddings in PyTorch .pt format for efficient loading and GPU compatibility
"""

import torch
import json
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2Model
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT2EmbeddingsPtSaver:
    def __init__(self, model_name="gpt2", device="auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.tokenizer = None
        self.model = None
        
    def _get_device(self, device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self):
        """Load GPT-2 model and tokenizer"""
        logger.info(f"Loading {self.model_name} model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2Model.from_pretrained(self.model_name)
        
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        # Move to device
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
    
    def save_embedding_matrices_pt(self, output_file="gpt2_embedding_matrices.pt"):
        """Save token and positional embedding matrices as PyTorch tensors"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Extract embedding matrices (keep as tensors)
        token_embeddings = self.model.wte.weight.detach().clone()
        positional_embeddings = self.model.wpe.weight.detach().clone()
        
        # Prepare data dictionary
        embedding_data = {
            'token_embeddings': token_embeddings,
            'positional_embeddings': positional_embeddings,
            'model_name': self.model_name,
            'vocab_size': len(self.tokenizer),
            'hidden_size': token_embeddings.shape[1],
            'max_position': positional_embeddings.shape[0],
            'device_saved': str(token_embeddings.device),
            'created_at': datetime.now().isoformat()
        }
        
        # Save as .pt file
        torch.save(embedding_data, output_file)
        logger.info(f"Saved embedding matrices to {output_file}")
        
        # Save vocabulary separately as JSON
        vocab_file = output_file.replace('.pt', '_vocab.json')
        vocab = self.tokenizer.get_vocab()
        with open(vocab_file, 'w') as f:
            json.dump(vocab, f, indent=2)
        
        return output_file, vocab_file
    
    def save_contextual_embeddings_pt(self, texts, output_file="gpt2_contextual.pt", max_length=512):
        """Save contextual embeddings for input texts as PyTorch tensors"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer not loaded.")
        
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            
        # Compute sentence embeddings with masking
        mask = attention_mask.unsqueeze(-1).float()
        masked_hidden = hidden_states * mask
        lengths = mask.sum(dim=1)
        sentence_embeddings = masked_hidden.sum(dim=1) / lengths
        
        # Prepare contextual data (keep as tensors)
        contextual_data = {
            'sentence_embeddings': sentence_embeddings.detach().clone(),
            'token_embeddings': hidden_states.detach().clone(),
            'attention_masks': attention_mask.detach().clone(),
            'input_ids': input_ids.detach().clone(),
            'texts': texts,
            'model_name': self.model_name,
            'max_length': max_length,
            'device_saved': str(hidden_states.device),
            'created_at': datetime.now().isoformat()
        }
        
        # Save as .pt file
        torch.save(contextual_data, output_file)
        logger.info(f"Saved contextual embeddings to {output_file}")
        
        return output_file
    
    def save_complete_embeddings_pt(self, texts=None, output_file="gpt2_complete_embeddings.pt"):
        """Save all embeddings (matrices + contextual) in one .pt file"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Get embedding matrices
        token_embeddings = self.model.wte.weight.detach().clone()
        positional_embeddings = self.model.wpe.weight.detach().clone()
        
        # Prepare complete data dictionary
        complete_data = {
            # Embedding matrices
            'token_embeddings': token_embeddings,
            'positional_embeddings': positional_embeddings,
            
            # Model metadata
            'model_name': self.model_name,
            'vocab_size': len(self.tokenizer),
            'hidden_size': token_embeddings.shape[1],
            'max_position': positional_embeddings.shape[0],
            'device_saved': str(token_embeddings.device),
            'created_at': datetime.now().isoformat(),
            
            # Vocabulary
            'vocabulary': self.tokenizer.get_vocab(),
        }
        
        # Add contextual embeddings if texts provided
        if texts:
            contextual_data = self._extract_contextual_tensors(texts)
            complete_data['contextual'] = contextual_data
        
        # Save everything in one .pt file
        torch.save(complete_data, output_file)
        logger.info(f"Saved complete embeddings to {output_file}")
        
        return output_file
    
    def _extract_contextual_tensors(self, texts, max_length=512):
        """Helper method to extract contextual embeddings as tensors"""
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            
        # Compute sentence embeddings
        mask = attention_mask.unsqueeze(-1).float()
        masked_hidden = hidden_states * mask
        lengths = mask.sum(dim=1)
        sentence_embeddings = masked_hidden.sum(dim=1) / lengths
        
        return {
            'sentence_embeddings': sentence_embeddings.detach().clone(),
            'token_embeddings': hidden_states.detach().clone(),
            'attention_masks': attention_mask.detach().clone(),
            'input_ids': input_ids.detach().clone(),
            'texts': texts
        }

def load_pt_embeddings(file_path, device="cpu"):
    """
    Load embeddings from .pt file
    
    Args:
        file_path: Path to .pt file
        device: Device to load tensors to ("cpu", "cuda", etc.)
    
    Returns:
        Dictionary with loaded embeddings
    """
    # Load the saved data
    data = torch.load(file_path, map_location=device)
    
    logger.info(f"Loaded embeddings from {file_path}")
    logger.info(f"Model: {data.get('model_name', 'unknown')}")
    logger.info(f"Vocab size: {data.get('vocab_size', 'unknown')}")
    logger.info(f"Hidden size: {data.get('hidden_size', 'unknown')}")
    
    if 'contextual' in data:
        logger.info(f"Contextual embeddings for {len(data['contextual']['texts'])} texts")
    
    return data

def quick_save_embeddings(model_name="gpt2", texts=None, output_file="embeddings.pt"):
    """Quick function to save embeddings in one line"""
    saver = GPT2EmbeddingsPtSaver(model_name)
    saver.load_model()
    return saver.save_complete_embeddings_pt(texts, output_file)

def main():
    # Example usage
    saver = GPT2EmbeddingsPtSaver(model_name="gpt2")
    saver.load_model()
    
    print("Saving GPT-2 embeddings in PyTorch .pt format...")
    
    complete_file = saver.save_complete_embeddings_pt("checkpoints/gpt2_complete.pt")
    print(f"Saved complete: {complete_file}")

if __name__ == "__main__":
    main()