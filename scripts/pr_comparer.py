import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import transformers
transformers.logging.set_verbosity_error()  # Suppress model warnings
import logging
import json
import csv
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics.distance import edit_distance
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ------------------------- Original Functions (Enhanced) -------------------------
def load_json(file_path):
    """Loads JSON data with detailed error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            logger.debug(f"Successfully loaded: {file_path}")
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {file_path}\nError: {e}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Unexpected error loading {file_path}:\n{e}")
    return None

def find_comparison_pairs(root_dir="experiment/prs"):
    """Generate comparison pairs between baseline and LLM versions"""
    VERSION_MAP = {
        'degraded': {'baseline_file': 'degraded_pr.json', 'llm_file': 'degraded_pr.json'},
        'improved_degraded': {'baseline_file': 'improved_degraded.json', 'llm_file': 'improved_degraded_pr.json'},
        'improved_original': {'baseline_file': 'improved_original_pr.json', 'llm_file': 'improved_original_pr.json'}
    }

    pairs = []
    logger.info(f"Scanning directory for PR entries: {root_dir}")
    
    for pr_entry in os.listdir(root_dir):
        if pr_entry == "ENTRY TEMPLATE":
            continue

        try:
            pr_num = int(pr_entry.split()[0])
            if 1 <= pr_num <= 6:
                pr_path = os.path.join(root_dir, pr_entry)
                other_llms_dir = next((os.path.join(pr_path, d) for d in os.listdir(pr_path) 
                                      if d.startswith("other_llms")), None)
                
                if other_llms_dir and os.path.exists(other_llms_dir):
                    for version, files in VERSION_MAP.items():
                        baseline_dir = os.path.join(pr_path, "versions", version)
                        baseline_path = os.path.join(baseline_dir, files['baseline_file'])
                        
                        if os.path.exists(baseline_path):
                            for llm in os.listdir(other_llms_dir):
                                version_path = os.path.join(other_llms_dir, llm, "versions", 
                                                          version, files['llm_file'])
                                if os.path.exists(version_path):
                                    pairs.append((baseline_path, version_path, version, llm, pr_num))
                                    logger.debug(f"Found pair: PR{pr_num} {version} vs {llm}")
        except (ValueError, IndexError) as e:
            logger.warning(f"Skipping invalid entry: {pr_entry} ({e})")
            continue
            
    logger.info(f"Found {len(pairs)} comparison pairs")
    return pairs

# ------------------------- Enhanced Comparison Class -------------------------
class PRComparator:
    def __init__(self):
        logger.info("Initializing PRComparator with ML models")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        logger.debug("Models initialized successfully")

    def extract_structured_text(self, data):
        """Extract and structure text components with metadata"""
        logger.debug("Extracting structured text components")
        return {
            'title': data.get('title', ''),
            'body': data.get('body', ''),
            'comments': [c['body'] for c in data.get('comments', {}).get('nodes', []) if c.get('body', '').strip()],
            'reviews': [r['body'] for r in data.get('reviews', {}).get('nodes', []) if r.get('body', '').strip()],
            'review_threads': [ct['body'] for rt in data.get('reviewThreads', {}).get('nodes', []) 
                             for ct in rt.get('comments', {}).get('nodes', []) if ct.get('body', '').strip()]
        }

    def calculate_lexical_similarity(self, baseline, llm):
        """Calculate traditional text similarity metrics"""
        logger.info("Calculating lexical similarity metrics (BLEU, Jaccard, Edit Distance)")
        baseline_flat = ' '.join([v if isinstance(v, str) else ' '.join(v) for v in baseline.values()])
        llm_flat = ' '.join([v if isinstance(v, str) else ' '.join(v) for v in llm.values()])
        return {
            'bleu': sentence_bleu([baseline_flat.split()], llm_flat.split(), smoothing_function=self.smoothing),
            'jaccard': len(set(baseline_flat.split()) & set(llm_flat.split())) / max(1, len(set(baseline_flat.split()) | set(llm_flat.split()))),
            'edit_distance': edit_distance(baseline_flat, llm_flat) / max(len(baseline_flat), len(llm_flat), 1)
        }

    def calculate_semantic_similarity(self, baseline, llm):
        """Calculate embedding-based similarity metrics"""
        logger.info("Calculating semantic similarity (SBERT, BERTScore)")
        texts = [
            ' '.join([v if isinstance(v, str) else ' '.join(v) for v in baseline.values()]),
            ' '.join([v if isinstance(v, str) else ' '.join(v) for v in llm.values()])
        ]
        sbert_embeds = self.sbert_model.encode(texts)
        P, R, F1 = bert_score([texts[1]], [texts[0]], lang='en')
        return {
            'sbert_cosine': sklearn_cosine([sbert_embeds[0]], [sbert_embeds[1]])[0][0],
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item()
        }

    def calculate_rouge_scores(self, baseline, llm):
        """Calculate ROUGE metrics for title and body"""
        logger.info("Calculating ROUGE scores for title/body")
        scores = {}
        for component in ['title', 'body']:
            base_text = baseline.get(component, '')
            llm_text = llm.get(component, '')
            scores[component] = self.rouge_scorer.score(base_text, llm_text)
        return scores

    def calculate_tfidf_similarity(self, baseline, llm):
        """Calculate TF-IDF vector similarity"""
        logger.info("Calculating TF-IDF cosine similarity")
        texts = [
            ' '.join([v if isinstance(v, str) else ' '.join(v) for v in baseline.values()]),
            ' '.join([v if isinstance(v, str) else ' '.join(v) for v in llm.values()])
        ]
        tfidf = TfidfVectorizer().fit_transform(texts)
        return sklearn_cosine(tfidf[0], tfidf[1])[0][0]

    def compare(self, baseline_data, llm_data):
        """Full comparison pipeline"""
        logger.info("Starting comparison pipeline")
        baseline = self.extract_structured_text(baseline_data)
        llm = self.extract_structured_text(llm_data)
        
        results = {}
        results.update(self.calculate_lexical_similarity(baseline, llm))
        results.update(self.calculate_semantic_similarity(baseline, llm))
        results['tfidf_cosine'] = self.calculate_tfidf_similarity(baseline, llm)
        
        rouge_scores = self.calculate_rouge_scores(baseline, llm)
        for comp in rouge_scores:
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                results[f"{comp}_{metric}"] = rouge_scores[comp][metric].fmeasure
                
        logger.info("Comparison completed successfully")
        return results

# ------------------------- Enhanced Main Execution -------------------------
def main():
    logger.info("====== Starting PR Comparison Analysis ======")
    comparator = PRComparator()
    comparison_pairs = find_comparison_pairs()
    
    if not comparison_pairs:
        logger.warning("No comparison pairs found! Exiting.")
        return

    logger.info(f"Processing {len(comparison_pairs)} comparison pairs")
    results = []
    
    for idx, (baseline_path, llm_path, version, llm, pr_num) in enumerate(comparison_pairs, 1):
        logger.info(f"\nProcessing pair {idx}/{len(comparison_pairs)}:")
        logger.info(f"PR Number: {pr_num}")
        logger.info(f"Version: {version.upper()}")
        logger.info(f"LLM: {llm}")
        logger.info(f"Baseline: {os.path.basename(baseline_path)}")
        logger.info(f"LLM File: {os.path.basename(llm_path)}")
        
        baseline_data = load_json(baseline_path)
        llm_data = load_json(llm_path)
        
        if baseline_data and llm_data:
            comparison_result = comparator.compare(baseline_data, llm_data)
            comparison_result.update({
                'base_pr': f"{pr_num}_{version}",
                'other_llm_version': f"{pr_num}_{llm}_{version}",
                'baseline_path': baseline_path,
                'llm_path': llm_path
            })
            results.append(comparison_result)
            logger.debug(f"Completed pair {idx}/{len(comparison_pairs)}")
        else:
            logger.warning(f"Skipping pair due to missing data: {baseline_path} vs {llm_path}")

    # Write results to CSV
    logger.info("\nFinalizing results...")
    fieldnames = [
        'base_pr', 'other_llm_version',
        'bleu', 'jaccard', 'edit_distance', 'sbert_cosine',
        'bertscore_precision', 'bertscore_recall', 'bertscore_f1',
        'title_rouge1', 'title_rouge2', 'title_rougeL',
        'body_rouge1', 'body_rouge2', 'body_rougeL', 'tfidf_cosine',
        'baseline_path', 'llm_path'
    ]
    
    with open('enhanced_pr_comparison.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"\nAnalysis complete. Results saved to enhanced_pr_comparison.csv")
    logger.info("====== Program Completed Successfully ======\n")

if __name__ == "__main__":
    main()