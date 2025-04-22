import json
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
from transformers import pipeline
import torch
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import gc
import torch._dynamo
import ijson
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import psutil  # Added for memory monitoring

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pr_scorer.log"),
        logging.StreamHandler()
    ]
)

torch._dynamo.config.suppress_errors = True

COMPETENCIES = {
    "Architecture Impact": "Evaluates architectural implications and design pattern changes",
    "Code Quality": "Assesses code correctness, error handling, and technical debt",
    "Collaboration": "Evaluates communication clarity and team collaboration aspects",
    "Maintainability": "Analyzes documentation, modularity, and long-term viability",
    "User Impact": "Assesses user-facing changes and UX considerations",
    "Technical Leadership": "Measures influence on technical direction and mentoring"
}

MODEL_NAME = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
INPUT_DIR = "pr_data"
OUTPUT_DIR = "analysis_results"
BATCH_SIZE = 64  # Reduced to help with memory
CHUNK_SIZE = 500  # Smaller chunks for better memory management
MAX_TEXT_LENGTH = 4096
BUFFER_SIZE = 500  # Reduced buffer size
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

class EnhancedPRProcessor:
    def __init__(self):
        self._log_memory("Before model loading")
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        
        if torch.cuda.is_available():
            model = model.to("cuda", dtype=TORCH_DTYPE)
            
        model = torch.compile(model, backend="eager")
        
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            framework="pt",
            batch_size=BATCH_SIZE,
            torch_dtype=TORCH_DTYPE
        )
        
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        self.results_buffer = []
        self._log_memory("After model initialization")
        logging.info(f"Using precision: {TORCH_DTYPE}")

    def _log_memory(self, context: str):
        """Log current memory usage"""
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 ** 3)  # GB
        logging.info(f"{context} - Memory usage: {mem:.2f}GB")

    def _text_pipeline(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'(```.*?```|`.*?`|<.*?>|\b\w+@\w+\.\w+\b|http\S+)', '', text, flags=re.DOTALL)
        return re.sub(r'\s+', ' ', text).strip()[:MAX_TEXT_LENGTH]

    def _process_batch(self, pr_batch: list, repo_name: str) -> pd.DataFrame:
        try:
            texts = [
                self._text_pipeline(f"{p.get('title','')} {p.get('body','')} {self._concat_comments(p)}")
                for p in pr_batch
            ]
            
            results = self.classifier(
                texts,
                candidate_labels=list(COMPETENCIES.keys()),
                hypothesis_template="In software development context, this change demonstrates {}",
                multi_label=True
            )
            
            batch_df = pd.DataFrame([{
                "repo": repo_name,
                "pr_number": p["number"],
                "created_at": pd.to_datetime(p["createdAt"]).isoformat(),
                "state": p["state"],
                "merged": p.get("mergedAt") is not None,
                "comments_count": p["comments"]["totalCount"],
                **{f"score_{label}": score 
                   for label, score in zip(res['labels'], res['scores'])}
            } for p, res in zip(pr_batch, results)])
            
            # Explicit cleanup
            del texts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return batch_df
        
        except Exception as e:
            logging.error(f"Batch processing failed: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def _concat_comments(self, pr: dict) -> str:
        return ' '.join(c['body'] for c in pr.get('comments', {}).get('nodes', []))

    def _write_results(self, df: pd.DataFrame, output_file: Path, force: bool = False):
        """Buffer results and write periodically to reduce I/O operations"""
        self.results_buffer.append(df)
        
        if len(self.results_buffer) >= BUFFER_SIZE or force:
            combined_df = pd.concat(self.results_buffer, ignore_index=True)
            logging.info(f"Writing {len(combined_df)} records to {output_file}")
            
            combined_df.to_csv(
                output_file,
                mode='a',
                header=not output_file.exists(),
                index=False
            )
            self.results_buffer = []
            
            # Cleanup
            del combined_df
            gc.collect()
            self._log_memory("After writing results")

            if force:
                self._temporal_analysis(df, output_file.stem.replace('_analysis', ''))

    def _temporal_analysis(self, df: pd.DataFrame, repo_name: str):
        try:
            temporal_df = df.copy()
            temporal_df['date'] = pd.to_datetime(temporal_df['created_at'])
            temporal_df.set_index('date', inplace=True)
            
            trends = temporal_df.filter(like='score_').resample('M').mean()
            
            trend_file = Path(OUTPUT_DIR) / f"{repo_name}_trends.csv"
            trends.to_csv(trend_file)
            logging.info(f"Saved temporal trends to {trend_file}")
        except Exception as e:
            logging.error(f"Temporal analysis failed: {str(e)}")

    def process_repo_file(self, file_path: Path):
        """Process repository file using streaming JSON parser"""
        repo_name = file_path.stem.replace('_prs', '')
        output_file = Path(OUTPUT_DIR) / f"{repo_name}_analysis.csv"
        logging.info(f"🚀 Starting processing for {repo_name} ({file_path.name})")
        self._log_memory("Start of file processing")
        
        current_batch = []
        total_prs = 0
        chunk_counter = 0
        
        with open(file_path, 'rb') as f:
            parser = ijson.items(f, 'item')
            
            with tqdm(desc=f"📂 {repo_name}", unit="PR", dynamic_ncols=True) as pbar:
                for pr in parser:
                    current_batch.append(pr)
                    total_prs += 1
                    pbar.update(1)
                    
                    if len(current_batch) >= CHUNK_SIZE:
                        self._process_chunk(current_batch, repo_name, output_file, chunk_counter)
                        chunk_counter += 1
                        current_batch = []
                        pbar.set_postfix({
                            "Chunks": chunk_counter, 
                            "PRs": total_prs,
                            "Mem": f"{psutil.Process().memory_info().rss/(1024**2):.1f}MB"
                        })
                        self._log_memory(f"After chunk {chunk_counter}")
                
                # Process remaining PRs
                if current_batch:
                    self._process_chunk(current_batch, repo_name, output_file, chunk_counter)
                    self._log_memory("Final chunk processed")

        logging.info(f"✅ Completed {total_prs} PRs from {repo_name}")
        self._write_results(pd.DataFrame(), output_file, force=True)  # Final flush
        self._log_memory("End of file processing")

    def _process_chunk(self, chunk: list, repo_name: str, output_file: Path, chunk_number: int):
        """Helper method to process a chunk of PRs"""
        logging.info(f"🔧 Processing chunk {chunk_number} ({len(chunk)} PRs)")
        with ThreadPoolExecutor(max_workers=2) as executor:  # Reduced workers
            futures = [
                executor.submit(self._process_batch, 
                             chunk[i:i+BATCH_SIZE], 
                             repo_name)
                for i in range(0, len(chunk), BATCH_SIZE)
            ]
            
            with tqdm(total=len(futures), desc=f"⚙️  {repo_name} batches", leave=False) as batch_pbar:
                for future in futures:
                    chunk_df = future.result()
                    if not chunk_df.empty:
                        self._write_results(chunk_df, output_file, force=(chunk_number == 0))
                    del chunk_df
                    batch_pbar.update(1)
                    self._log_memory("After batch processing")

def main():
    processor = EnhancedPRProcessor()
    pr_files = list(Path(INPUT_DIR).glob("*.json"))
    
    logging.info(f"Found {len(pr_files)} repository files to process")
    
    for pr_file in tqdm(pr_files, desc="📁 Repositories", unit="repo"):
        logging.info(f"\n{'='*40}\nStarting processing for {pr_file.name}\n{'='*40}")
        processor.process_repo_file(pr_file)
        processor._log_memory("After repository processing")

if __name__ == "__main__":
    main()