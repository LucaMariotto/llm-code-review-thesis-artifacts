#!/usr/bin/env python3
"""
Improved PR Fetcher Script for GitHub GraphQL API with Multiple Repository Support
"""

import datetime
import json
import logging
import os
import random
import time
from time import sleep

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Hardcoded GitHub tokens (replace these with your actual tokens)
GITHUB_TOKENS = [
    #250 TOKENS USED
]

REPOS = [
    # Programming Languages
    #('torvalds', 'linux'),                   # Linux kernel: high-quality, strict review process
    # ('rust-lang', 'rust'),                   # Rust language: rigorous safety and performance focus
    # ('golang', 'go'),                        # Go language: strict discipline in review practices
    # ('apple', 'swift'),                      # Swift language: well-structured and high-quality PR reviews
    # ('llvm', 'llvm-project'),                # LLVM compiler infrastructure with top-tier code quality
    # ('microsoft', 'TypeScript'),             # TypeScript: strong type safety and disciplined review process
    # ('python', 'cpython'),                   # CPython: reference implementation of Python with strict code standards
    # ('openjdk', 'jdk'),                      # OpenJDK: Java Development Kit with a formalized review process

    # Machine Learning & AI
    #('tensorflow', 'tensorflow'),            # Industry-standard ML framework
    # ('pytorch', 'pytorch'),                  # Deep learning library with strong engineering rigor
    # ('huggingface', 'transformers'),         # NLP library with structured contributions
    # ('scikit-learn', 'scikit-learn'),         # Machine learning library with in-depth reviews
    # ('facebookresearch', 'detectron2'),      # Object detection platform with high review standards
    # ('fastai', 'fastai'),                    # High-level deep learning library with community-driven reviews

    # Web & Frontend Frameworks
    #('facebook', 'react'),                   # React.js: high-engagement, quality code discussions
    # ('angular', 'angular'),                  # Angular framework with Google’s engineering standards
    # ('vercel', 'next.js'),                   # Next.js: widely used React-based framework
    # ('vuejs', 'vue'),                        # Vue.js: active and well-reviewed contributions
    # ('solidjs', 'solid'),                    # Solid.js: reactive frontend framework with strong reviews
    # ('sveltejs', 'svelte'),                  # Svelte: innovative UI framework with a focus on clarity
    # ('emberjs', 'ember.js'),                 # Ember.js: mature framework with well-organized PR discussions
    # ('tailwindlabs', 'tailwindcss'),         # Tailwind CSS: utility-first CSS framework with active community reviews
    # ('flutter', 'flutter'),                   # Flutter: Google's cross-platform UI toolkit with strict code quality

    # Backend & Cloud Infrastructure
    # #('kubernetes', 'kubernetes'),            # Cloud-native orchestration system with thorough PRs
    # ('hashicorp', 'terraform'),              # Infrastructure as code with strict validation
    # ('docker', 'cli'),                       # Docker's command-line tool, core container tech
    # ('moby', 'moby'),                        # Docker engine, key containerization tech
    # #('apache', 'spark'),                     # Big data processing framework with quality reviews
    # ('apache', 'kafka'),                     # Distributed event streaming platform
    # ('elastic', 'elasticsearch'),            # Enterprise search engine with high engineering standards
    # ('prometheus', 'prometheus'),            # Monitoring system with a strong review process
    # ('nodejs', 'node'),                      # Node.js: runtime with rigorous code reviews
    # ('apache', 'hadoop'),                    # Hadoop: distributed storage and processing with formal review protocols
    # ('apache', 'cassandra'),                 # Cassandra: scalable NoSQL database with established review practices
    # ('containerd', 'containerd'),                   # containerd: Industry-standard container runtime
    # ('istio', 'istio'),                       # Istio: Service mesh with enterprise-grade review processes
    # ('pandas-dev', 'pandas'),                # pandas: data manipulation library with rigorous review standards

    # Development Tools
    # #('microsoft', 'vscode'),                 # Widely used open-source editor with disciplined contribution practices
    # ('neovim', 'neovim'),                    # Modern Vim editor with strict code quality standards
    # ('eslint', 'eslint'),                    # Linter for JavaScript/TypeScript enforcing consistent style
    # ('prettier', 'prettier'),                # Opinionated code formatter with industry-wide adoption
    # ('git', 'git'),                          # Git version control system with community-driven maintenance
    # ('openresty', 'openresty'),              # High-performance web platform built on Nginx with rigorous reviews
    # ('bazelbuild', 'bazel'),                 # Bazel: Google's build system known for its rigorous code quality and disciplined review process


    # Databases & Storage
    # ('postgres', 'postgres'),                # PostgreSQL: enterprise-grade, rigorously reviewed database
    # ('redis', 'redis'),                      # In-memory database with structured contributions
    # ('mariadb', 'server'),                   # MariaDB: high-performance fork of MySQL with strong review processes
    # ('mongodb', 'mongo'),                     # MongoDB: NoSQL database with active contributor reviews
    # ('cockroachdb', 'cockroach'),            # Distributed SQL database with modern review practices
    # ('etcd-io', 'etcd'),                     # Distributed key-value store with disciplined PR discussions
    # ('facebook', 'rocksdb'),                 # RocksDB: high-performance key-value store with meticulous reviews
    # ('influxdata', 'influxdb'),              # Time series database with structured code quality checks

    # Security & Cryptography
    # ('openssl', 'openssl'),                  # Cryptography library with high scrutiny and rigorous reviews
    # ('hashicorp', 'vault'),                  # Secrets management with strong security standards
    # ('sigstore', 'cosign'),                  # Secure container signing framework with community oversight
    # ('letsencrypt', 'boulder'),              # ACME server for HTTPS certificates with strict review process
    # ('libressl', 'libressl'),                # Fork of OpenSSL focused on simplicity and security

    # Automation & DevOps
    # ('ansible', 'ansible'),                  # Configuration management tool with detailed peer reviews
    # ('puppetlabs', 'puppet'),                # Infrastructure automation with disciplined review processes
    # ('jenkinsci', 'jenkins'),                # Automation server with an active and strict review culture
    # ('actions', 'runner'),                   # GitHub Actions runner with enterprise-quality code reviews
    # ('grafana', 'grafana'),                  # Visualization and monitoring framework with strong engineering practices
    # ('saltstack', 'salt'),                   # Configuration management tool known for rigorous quality control

    # Networking
    # ('cilium', 'cilium'),                    # eBPF-based networking and security with thorough PR reviews
    # ('zephyrproject-rtos', 'zephyr'),          # Embedded RTOS with strong engineering rigor
    # ('coredns', 'coredns'),                  # DNS server critical to Kubernetes with high-quality reviews

    # Operating Systems & Low-Level Software
    # ('systemd', 'systemd'),                  # Linux system and service manager with strict review process
    # ('qemu', 'qemu'),                        # Virtualization software with detailed code scrutiny
    # ('xen-project', 'xen'),                  # Hypervisor with enterprise-grade code practices
    # ('openbsd', 'src'),                      # OpenBSD source: renowned for security and careful reviews
    # ('freebsd', 'freebsd'),                  # FreeBSD OS: long-standing rigorous review standards
    # ('reactos', 'reactos'),                  # Open-source operating system with community-driven code reviews

    # Web & API Frameworks
    #('django', 'django'),                    # Python web framework with extensive and well-documented reviews
    # ('pallets', 'flask'),                      # Lightweight web framework with clear review processes
    # ('expressjs', 'express'),                # Node.js web framework with active review and contribution history
    # ('fastify', 'fastify'),                  # High-performance Node.js framework with disciplined PRs
    # ('spring-projects', 'spring-framework'),   # Java framework with rigorous code review practices
    # ('rails', 'rails'),                       # Ruby on Rails: mature full-stack framework with a strong review culture
    # ('aspnet', 'aspnetcore'),                # Cross-platform web framework with strict quality controls

    # Game Engines & Graphics
    # ('godotengine', 'godot'),                # Open-source game engine with active community and quality reviews
    # ('pygame', 'pygame'),                    # Game development library for Python with clear coding standards
    # ('blender', 'blender'),                  # 3D creation suite with extensive code contributions and reviews
    # ('defold', 'defold'),                    # Open-source game engine with high industry standards
    # ('openframeworks', 'openFrameworks'),    # Toolkit for creative coding with disciplined code reviews
]



# Configuration parameters
START_DATE_STR = "2022-01-01"
END_DATE_STR = "2025-01-01"
MIN_COMMENTS = 5
MAX_COMMENTS = 15
PAGE_SIZE = 15

# Constants for rate limiting
TOKEN_REMAINING_THRESHOLD = 500
INITIAL_BACKOFF = 1
SKIP_TIMEOUT = 60
EXTENDED_BACKOFF_THRESHOLD = 5
EXTENDED_BACKOFF_DELAY = 60

# GraphQL query to fetch pull request data
QUERY = """
query ($owner: String!, $name: String!, $pageSize: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(
      states: [CLOSED, MERGED]
      first: $pageSize
      after: $after
      orderBy: {field: CREATED_AT, direction: DESC}
    ) {
      edges {
        node {
          number
          title
          body
          state
          createdAt
          closedAt
          mergedAt
          url
          author {
            login
            url
          }
          comments(first: 15) {
            totalCount
            nodes {
              body
              author {
                login
                url
              }
              createdAt
              reactions(first: 10) {
                nodes {
                  content
                }
              }
            }
          }
          reviews(first: 10) {
            totalCount
            nodes {
              body
              author {
                login
                url
              }
              state
              createdAt
            }
          }
          reviewThreads(first: 10) {
            totalCount
            nodes {
              comments(first: 10) {
                nodes {
                  body
                  author {
                    login
                    url
                  }
                  createdAt
                }
              }
            }
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
      totalCount
    }
  }
}
"""

def get_total_comments(pr):
    """Calculate total number of comments for a pull request"""
    comments = pr.get('comments', {}).get('totalCount', 0)
    reviews = pr.get('reviews', {}).get('totalCount', 0)
    review_threads = pr.get('reviewThreads', {}).get('totalCount', 0)
    return comments + reviews + review_threads

def get_github_session(skip_tokens):
    """Get a GitHub session with token rotation and rate limit checking"""
    current_time = time.time()
    # Clean up expired skip entries using reset times
    expired_tokens = [token for token, reset in skip_tokens.items() if current_time > reset]
    for token in expired_tokens:
        del skip_tokens[token]

    available_tokens = [token for token in GITHUB_TOKENS if token not in skip_tokens]
    
    if not available_tokens:
        earliest_reset = min(skip_tokens.values()) if skip_tokens else current_time
        wait_time = max(earliest_reset - current_time, 0)
        logging.warning(f"All tokens rate-limited. Waiting {wait_time:.0f} seconds.")
        sleep(wait_time + 1)
        return get_github_session(skip_tokens)
    
    random.shuffle(available_tokens)
    
    for token in available_tokens:
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        })
        session.token = token
        rate_limit_url = "https://api.github.com/rate_limit"
        try:
            response = session.get(rate_limit_url)
            if response.status_code != 200:
                continue
                
            rate_limit_data = response.json()
            graphql_limit = rate_limit_data.get('resources', {}).get('graphql', {})
            remaining = graphql_limit.get('remaining', 0)
            reset_time = graphql_limit.get('reset', 0)

            if remaining > TOKEN_REMAINING_THRESHOLD:
                return session
            else:
                logging.info(f"Token ...{token[-4:]} near limit (Remaining: {remaining}). Will skip until {reset_time}")
                skip_tokens[token] = reset_time  # Mark to skip until reset time
                continue  # Try next token immediately

        except Exception as e:
            logging.error(f"Error checking rate limit for token ...{token[-4:]}: {e}")
            skip_tokens[token] = current_time + SKIP_TIMEOUT
            continue

    logging.warning("No available tokens. Waiting 60 seconds.")
    sleep(60)
    return get_github_session(skip_tokens)

def fetch_pr_page(after_cursor, page_size, owner, name):
    """Fetch a page of pull requests for a specific repository"""
    skip_tokens = {}
    attempt = 0
    consecutive_failures = 0
    
    while True:
        attempt += 1
        session = get_github_session(skip_tokens)
        variables = {
            "owner": owner,
            "name": name,
            "pageSize": page_size,
            "after": after_cursor
        }
        
        logging.info(f"Fetching {owner}/{name} page with cursor: {after_cursor} using token ...{session.token[-4:]}")
        
        try:
            response = session.post("https://api.github.com/graphql", json={'query': QUERY, 'variables': variables})
        except requests.exceptions.ChunkedEncodingError as e:
            logging.error(f"ChunkedEncodingError with token ...{session.token[-4:]}: {e}")
            skip_tokens[session.token] = time.time()
            sleep(2)
            consecutive_failures += 1
            continue
        except Exception as e:
            logging.error(f"Unexpected error with token ...{session.token[-4:]}: {e}")
            skip_tokens[session.token] = time.time()
            sleep(2)
            consecutive_failures += 1
            continue

        status = response.status_code

        if status in [502, 504]:
            logging.warning(f"Server error ({status}) with token ...{session.token[-4:]}. Retrying.")
            skip_tokens[session.token] = time.time()
            consecutive_failures += 1
            backoff = min(INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1), 60)
            sleep(backoff)
        elif status == 403:
            reset_time = int(response.headers.get('x-ratelimit-reset', time.time()+60))
            wait_time = max(reset_time - time.time(), 0)
            logging.warning(f"Rate limited with token ...{session.token[-4:]}. Waiting {wait_time:.0f}s.")
            sleep(wait_time + 1)
            consecutive_failures += 1
        elif status != 200:
            logging.error(f"Unexpected status {status} with token ...{session.token[-4:]}: {response.text}")
            skip_tokens[session.token] = time.time()
            sleep(5)
            consecutive_failures += 1
        else:
            try:
                result = response.json()
            except Exception as e:
                logging.error(f"JSON decode error with token ...{session.token[-4:]}: {e}")
                skip_tokens[session.token] = time.time()
                sleep(5)
                consecutive_failures += 1
                continue

            if "errors" in result:
                logging.error(f"GraphQL error with token ...{session.token[-4:]}: {result['errors']}")
                skip_tokens[session.token] = time.time()
                sleep(5)
                consecutive_failures += 1
                continue

            points_used = int(response.headers.get('x-ratelimit-used', 0))
            logging.info(f"Used {points_used} points with token ...{session.token[-4:]}")
            pr_data = result.get('data', {}).get('repository', {}).get('pullRequests', {})
            return pr_data, points_used

        if consecutive_failures >= EXTENDED_BACKOFF_THRESHOLD:
            logging.warning(f"{consecutive_failures} failures. Extended backoff {EXTENDED_BACKOFF_DELAY}s.")
            sleep(EXTENDED_BACKOFF_DELAY)
            consecutive_failures = 0

def save_progress(cursor, fetched_prs, owner, name):
    """Save progress for a specific repository"""
    progress_file = f"progress_{owner}_{name}.json"
    progress_data = {"cursor": cursor, "fetched_prs": fetched_prs}
    try:
        with open(progress_file, "w") as f:
            json.dump(progress_data, f, indent=2)
        logging.info(f"Progress saved to {progress_file}")
    except IOError as e:
        logging.error(f"Error writing to {progress_file}: {e}")

def load_progress(owner, name):
    """Load progress for a specific repository"""
    progress_file = f"progress_{owner}_{name}.json"
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error reading {progress_file}: {e}")
    return {"cursor": None, "fetched_prs": []}

def fetch_pull_requests(owner, name):
    """Main PR fetching logic for a single repository"""
    DATE_FORMAT = "%Y-%m-%d"
    START_DATE = datetime.datetime.strptime(START_DATE_STR, DATE_FORMAT)
    END_DATE = datetime.datetime.strptime(END_DATE_STR, DATE_FORMAT)
    
    progress = load_progress(owner, name)
    after_cursor = progress.get("cursor")
    fetched_prs = progress.get("fetched_prs", [])
    matching_prs = fetched_prs.copy()
    
    finished = False
    total_processed = 0
    date_filtered = 0
    comment_filtered = 0
    api_call_count = 0
    total_points_used = 0
    
    logging.info(f"Fetching PRs from {owner}/{name} ({START_DATE_STR} to {END_DATE_STR})")
    
    while not finished:
        pr_data, points_used = fetch_pr_page(after_cursor, PAGE_SIZE, owner, name)
        api_call_count += 1
        total_points_used += points_used
        
        if not pr_data:
            logging.error("Failed to fetch PR page. Exiting loop.")
            break
        
        edges = pr_data.get('edges', [])
        page_info = pr_data.get('pageInfo', {})
        
        if not edges:
            logging.info("No more PRs to process.")
            break
        
        for edge in edges:
            pr = edge.get('node', {})
            total_processed += 1
            try:
                pr_created = datetime.datetime.strptime(pr['createdAt'], '%Y-%m-%dT%H:%M:%SZ')
            except ValueError as e:
                logging.error(f"Date error PR #{pr.get('number', 'unknown')}: {e}")
                continue
            
            if pr_created > END_DATE:
                continue
            if pr_created < START_DATE:
                finished = True
                break
            
            date_filtered += 1
            total_comments = get_total_comments(pr)
            if MIN_COMMENTS <= total_comments <= MAX_COMMENTS:
                comment_filtered += 1
                pr['filesChanged'] = pr.get('files', {}).get('nodes', [])
                pr['comments']['nodes'] = pr.get('comments', {}).get('nodes', [])
                pr['reviews']['nodes'] = pr.get('reviews', {}).get('nodes', [])
                pr['reviewThreads']['nodes'] = pr.get('reviewThreads', {}).get('nodes', [])
                matching_prs.append(pr)
                logging.info(f"Match PR #{pr['number']}: {total_comments} comments")
            
            if total_processed % 100 == 0:
                logging.info(f"Progress: {total_processed} PRs | In date: {date_filtered} | Matched: {comment_filtered}")
        
        if not finished and page_info.get('hasNextPage'):
            after_cursor = page_info.get('endCursor')
            save_progress(after_cursor, matching_prs, owner, name)
            sleep(random.uniform(5, 10))
        else:
            finished = True
    
    logging.info(f"Final for {owner}/{name}:")
    logging.info(f"Total processed: {total_processed}")
    logging.info(f"In date range: {date_filtered}")
    logging.info(f"Matching PRs: {comment_filtered}")
    logging.info(f"API calls: {api_call_count}")
    logging.info(f"Total points used: {total_points_used}")
    
    return matching_prs, f"{owner}_{name}_prs.json"

def main():
    """Process all repositories sequentially"""
    for repo in REPOS:
        owner, name = repo
        logging.info(f"\n{'#'*40}")
        logging.info(f"Starting processing: {owner}/{name}")
        logging.info(f"{'#'*40}")
        
        prs_in_range, output_file = fetch_pull_requests(owner, name)
        
        try:
            with open(output_file, "w", encoding="utf-8") as outfile:
                json.dump(prs_in_range, outfile, indent=2)
            logging.info(f"Data saved to {output_file}")
        except IOError as e:
            logging.error(f"Failed to write {output_file}: {e}")
        
        logging.info(f"Completed processing {owner}/{name}\n")

if __name__ == "__main__":
    main()