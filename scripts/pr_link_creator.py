import csv

def generate_pr_links(csv_content):
    """
    Generate pull request links along with categories from CSV content.
    The CSV is expected to have the following columns:
      - 'repo': in the format "owner_repoName"
      - 'pr_number': the pull request number
      - 'category': the category associated with the PR
    This function returns a list of strings formatted as:
      <PR_URL> | <category>
    """
    # Split the CSV content into lines and create a CSV DictReader.
    lines = csv_content.strip().split('\n')
    reader = csv.DictReader(lines)
    
    results = []
    for row in reader:
        # Ensure that the required fields are present.
        if not all(key in row for key in ['repo', 'pr_number', 'category']):
            continue  # Skip rows missing required data.
        
        # Parse the 'repo' field to extract owner and repository name.
        repo_parts = row['repo'].split('_', 1)
        if len(repo_parts) != 2:
            continue  # Skip rows with an invalid repo format.
        owner, repo_name = repo_parts
        
        pr_number = row['pr_number']
        pr_url = f"https://github.com/{owner}/{repo_name}/pull/{pr_number}"
        category = row['category']
        
        # Combine the PR URL and category with the separator " | "
        link_with_category = f"{pr_url} | {category}"
        results.append(link_with_category)
    
    return results

if __name__ == '__main__':
    # Specify the CSV filename.
    csv_filename = "top_prs_per_category.csv"
    
    # Open and read the CSV file content.
    with open(csv_filename, "r") as csv_file:
        csv_content = csv_file.read()

    # Generate the pull request links with associated categories.
    pr_links = generate_pr_links(csv_content)

    # Print each link with its category.
    for link in pr_links:
        print(link)

    # Optionally, save the results to a text file.
    output_filename = "pr_links.txt"
    with open(output_filename, "w") as out_file:
        out_file.write("\n".join(pr_links))
