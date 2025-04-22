import heapq
import pandas as pd
from pathlib import Path

# Initialize heaps for each category and a global counter to break ties
heaps = {}
counter = 0

def process_csv_file(csv_file):
    global counter
    df = pd.read_csv(csv_file)
    score_columns = [col for col in df.columns if col.startswith('score_')]
    
    # Initialize heaps for new categories
    for col in score_columns:
        category = col.replace('score_', '').strip()
        if category not in heaps:
            heaps[category] = []
    
    # Process each row
    for _, row in df.iterrows():
        pr_data = {
            'repo': row['repo'],
            'pr_number': row['pr_number'],
            'created_at': row['created_at'],
            'state': row['state'],
            'merged': row['merged'],
            'comments_count': row['comments_count']
        }
        
        for col in score_columns:
            category = col.replace('score_', '').strip()
            score = row[col]
            if pd.isna(score):
                continue
            try:
                score = float(score)
            except ValueError:
                continue
            
            heap = heaps[category]
            entry = (score, counter, pr_data.copy())
            counter += 1  # Ensure unique counter for tie-breaking
            
            if len(heap) < 100:
                heapq.heappush(heap, entry)
            else:
                # Only add if new score > smallest score in the heap
                if score > heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, entry)

# Process all CSV files
for csv_file in Path('analysis_results\scored_repos').glob('*.csv'):
    process_csv_file(csv_file)

# Extract and sort the top 100 entries per category
csv_rows = []
for category in heaps:
    heap = heaps[category]
    # Sort by score descending, then by creation time ascending
    sorted_prs = sorted(heap, key=lambda x: (-x[0], x[2]['created_at']))
    for entry in sorted_prs:
        score, _, pr_data = entry
        row = {'category': category, 'score': score}
        row.update(pr_data)
        csv_rows.append(row)

# Save to CSV
df_output = pd.DataFrame(csv_rows)
df_output.to_csv('top_prs_per_category.csv', index=False)
print("Top 100 PRs per category saved successfully.")