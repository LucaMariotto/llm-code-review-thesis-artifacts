import os
import json
import csv
import logging
from textstat import flesch_reading_ease, flesch_kincaid_grade
from collections import defaultdict
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class PRQualityAnalyzer:
    def __init__(self):
        nltk.download([
            'vader_lexicon',
            'stopwords',
            'punkt',
            'punkt_tab'
        ], quiet=True)
        
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        logger.info("PRQualityAnalyzer initialized with NLP models")

    def extract_text_components(self, data):
        return {
            'title': data.get('title', ''),
            'body': data.get('body', ''),
            'comments': [c['body'] for c in data.get('comments', {}).get('nodes', [])],
            'reviews': [r['body'] for r in data.get('reviews', {}).get('nodes', [])],
            'review_threads': [ct['body'] for rt in data.get('reviewThreads', {}).get('nodes', [])
                             for ct in rt.get('comments', {}).get('nodes', [])]
        }

    def calculate_readability(self, text):
        return {
            'flesch_reading_ease': flesch_reading_ease(text),
            'flesch_kincaid_grade': flesch_kincaid_grade(text)
        }

    def calculate_sentiment(self, text):
        scores = self.sia.polarity_scores(text)
        return {
            'neg': scores['neg'],
            'neu': scores['neu'],
            'pos': scores['pos'],
            'compound': scores['compound']
        }

    def calculate_text_stats(self, text):
        words = nltk.word_tokenize(text)
        sentences = nltk.sent_tokenize(text)
        unique_words = set(word.lower() for word in words if word.isalpha())
        stopword_count = sum(1 for word in words if word.lower() in self.stop_words)
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'unique_words': len(unique_words),
            'avg_word_length': sum(len(word) for word in words)/len(words) if words else 0,
            'stopword_ratio': stopword_count/len(words) if words else 0
        }

    def analyze_component(self, component_name, text):
        full_text = ' '.join(text) if isinstance(text, list) else text
        metrics = {}
        
        # Readability metrics
        metrics.update({f"{component_name}_readability_{k}": v 
                      for k, v in self.calculate_readability(full_text).items()})
        
        # Sentiment analysis
        metrics.update({f"{component_name}_sentiment_{k}": v 
                      for k, v in self.calculate_sentiment(full_text).items()})
        
        # Text statistics
        metrics.update({f"{component_name}_{k}": v 
                      for k, v in self.calculate_text_stats(full_text).items()})
        
        return metrics

    def analyze_pr(self, data):
        components = self.extract_text_components(data)
        metrics = {}
        
        for component, text in components.items():
            if component in ['comments', 'reviews', 'review_threads']:
                combined_text = ' '.join(text)
                metrics.update(self.analyze_component(component, combined_text))
            else:
                metrics.update(self.analyze_component(component, text))
        
        # Overall analysis
        all_text = ' '.join([str(v) for v in components.values()])
        metrics.update(self.analyze_component('overall', all_text))
        
        return metrics

def find_pr_files(root_dir="experiment/prs"):
    """Find all PR files in the directory structure"""
    file_map = defaultdict(list)
    logger.info(f"Scanning directory for PR files: {root_dir}")

    for pr_entry in os.listdir(root_dir):
        if pr_entry == "ENTRY TEMPLATE":
            continue

        try:
            pr_num = int(pr_entry.split()[0])
            pr_path = os.path.join(root_dir, pr_entry)
            
            # Process baseline versions
            versions_dir = os.path.join(pr_path, "versions")
            if os.path.exists(versions_dir):
                for version in os.listdir(versions_dir):
                    version_dir = os.path.join(versions_dir, version)
                    json_files = [f for f in os.listdir(version_dir) if f.endswith('.json')]
                    for fname in json_files:
                        file_map[pr_num].append({
                            'type': 'baseline',
                            'pr_number': pr_num,
                            'version': version,
                            'path': os.path.join(version_dir, fname)
                        })

            # Process LLM versions - MODIFIED SECTION
            other_llms_dir = os.path.join(pr_path, "other_llms")
            if os.path.exists(other_llms_dir):
                for llm in os.listdir(other_llms_dir):
                    llm_dir = os.path.join(other_llms_dir, llm)
                    versions_dir = os.path.join(llm_dir, "versions")
                    if os.path.exists(versions_dir):
                        for version in os.listdir(versions_dir):
                            # Skip 'original' version for non-baseline LLMs
                            if version == "original":
                                continue
                            version_dir = os.path.join(versions_dir, version)
                            json_files = [f for f in os.listdir(version_dir) if f.endswith('.json')]
                            for fname in json_files:
                                file_map[pr_num].append({
                                    'type': llm,
                                    'pr_number': pr_num,
                                    'version': version,
                                    'path': os.path.join(version_dir, fname)
                                })

        except (ValueError, IndexError) as e:
            logger.warning(f"Skipping invalid entry: {pr_entry} ({e})")
    
    logger.info(f"Found {sum(len(v) for v in file_map.values())} PR files")
    return file_map

def load_pr_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None

def main():
    logger.info("====== Starting PR Quality Analysis ======")
    analyzer = PRQualityAnalyzer()
    pr_files = find_pr_files()
    
    if not pr_files:
        logger.warning("No PR files found! Exiting.")
        return

    results = []
    for pr_num, files in pr_files.items():
        logger.info(f"Processing PR {pr_num} with {len(files)} files")
        
        for file_info in files:
            pr_data = load_pr_data(file_info['path'])
            if not pr_data:
                continue
            
            metrics = analyzer.analyze_pr(pr_data)
            metrics.update({
                'type': file_info['type'],
                'pr_number': file_info['pr_number'],
                'version': file_info['version'],
                'file_path': file_info['path']
            })
            results.append(metrics)

    # Write results to CSV
    logger.info("\nWriting results to CSV...")
    
    # Dynamically build fieldnames
    base_components = ['title', 'body', 'comments', 'reviews', 'review_threads', 'overall']
    fieldnames = ['type', 'pr_number', 'version', 'file_path']

    for component in base_components:
        fieldnames.extend([
            f"{component}_readability_flesch_reading_ease",
            f"{component}_readability_flesch_kincaid_grade",
            f"{component}_sentiment_neg",
            f"{component}_sentiment_neu",
            f"{component}_sentiment_pos",
            f"{component}_sentiment_compound",
            f"{component}_word_count",
            f"{component}_sentence_count",
            f"{component}_unique_words",
            f"{component}_avg_word_length",
            f"{component}_stopword_ratio"
        ])

    with open('pr_quality_analysis.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"\nAnalysis complete. Results saved to pr_quality_analysis.csv")
    logger.info("====== Program Completed ======\n")

if __name__ == "__main__":
    main()