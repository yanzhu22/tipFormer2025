#!/usr/bin/env python3
"""
pubmed_filter.py
"""

import argparse
import re
import time
import logging
import xml.etree.ElementTree as ET
from multiprocessing.pool import ThreadPool
from typing import Union, List, Tuple, Optional

import pandas as pd
from Bio import Entrez
from tqdm import tqdm

# -----------------------------------------------------------
#
# -----------------------------------------------------------

# 
KEYWORDS = re.compile(
    r"\b(bind|binding|affinity|interact|interaction|active|"
    r"inhibitor|agonist|antagonist|activator|"
    r"blocker|antagonism|synergist|potentiator|"
    r"modulate|modulator|efficacy|IC50|EC50|Kd|Ki)\b",
    re.IGNORECASE
)

# 
RETMAX = 5  # 
MAX_RETRIES = 5  # 
INITIAL_DELAY = 1  # 
BACKOFF_FACTOR = 2  #
MAX_DELAY = 60  #


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# -----------------------------------------------------------

def safe_parse_xml(xml_data: str) -> Optional[ET.Element]:
    try:
        return ET.fromstring(xml_data)
    except ET.ParseError as e:
        logger.warning(f"XML error: {str(e)}")
        return None

def query_pubmed(term: str, email: str, api_key: Union[str, None] = None) -> bool:
    """
    Retrieve PubMed and check for keywords in abstracts
    Returns: True=Interacting keywords present, False=No interaction
    """
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    # Step1: 
    delay = INITIAL_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            #
            time.sleep(0.3) 
                      
            handle = Entrez.esearch(
                db="pubmed",
                term=term,
                retmax=RETMAX,
                sort="relevance", 
                usehistory="y"    
            )
            xml_data = handle.read()
            handle.close()
            

            root = safe_parse_xml(xml_data)
            if not root:
                continue
                
            id_list = root.find(".//IdList")
            if id_list is None:
                logger.debug(f"IdList not found: {xml_data}")
                return False
                
            ids = [id_elem.text for id_elem in id_list.findall("Id")]
            if not ids:
                return False
            break
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:

                sleep_time = min(delay, MAX_DELAY)
                logger.warning(f"Esearch error (try {attempt+1}/{MAX_RETRIES}): {str(e)}. Retry after {sleep_time} seconds....")
                time.sleep(sleep_time)
                delay *= BACKOFF_FACTOR
            else:
                logger.error(f"Esearch error: {str(e)}")
                return False  

    # Step 2
    delay = INITIAL_DELAY
    for attempt in range(MAX_RETRIES):
        try:
           
            time.sleep(0.3)
            
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(ids),
                rettype="medline",
                retmode="text",
                retmax=RETMAX
            )
            data = handle.read()
            handle.close()
            
          
            if KEYWORDS.search(data):
                return True
            return False
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
 
                sleep_time = min(delay, MAX_DELAY)
                logger.warning(f"Efetch error (try {attempt+1}/{MAX_RETRIES}): {str(e)}. {sleep_time} seconds retry...")
                time.sleep(sleep_time)
                delay *= BACKOFF_FACTOR
            else:
                logger.error(f"Efetch failed: {str(e)}")
                return False 


def build_term(gene_name: str, uniprot_id: str, chem_name: str) -> str:
    """
        (gene_name OR uniprot_id) AND chem_name
    """

    gene_name = gene_name.strip() if gene_name else ""
    uniprot_id = uniprot_id.strip() if uniprot_id else ""
    chem_name = chem_name.strip() if chem_name else ""
    
    protein_terms = []
    if gene_name:
        protein_terms.append(f'"{gene_name}"[Title/Abstract]')
    if uniprot_id:
        protein_terms.append(f'"{uniprot_id}"[Title/Abstract]')
    
    chem_terms = []
    if chem_name:
      
        clean_chem = re.sub(r'[^\w\s-]', '', chem_name)
        chem_terms.append(f'"{clean_chem}"[Title/Abstract]')
    
    if not protein_terms or not chem_terms:
        logger.debug(f"invalid search: protein={protein_terms}, chem={chem_terms}")
        return ""
    
    protein_query = " OR ".join(protein_terms)
    chem_query = " OR ".join(chem_terms)
    
    return f"({protein_query}) AND ({chem_query})"


def worker(args: Tuple) -> Tuple[int, bool]:

    row, email, api_key = args
    try:
        term = build_term(
            row.GeneName,      # GeneName
            row.UniProtKB,     
            row.ChemicalName   
        )
        
        if not term:
            return row.name, False  
        
        return row.name, query_pubmed(term, email, api_key)
    except Exception as e:
        logger.error(f"row {row.name}: {str(e)}")
        return row.name, False


# -----------------------------------------------------------

# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PubMed-based negative sample filter")
    parser.add_argument("--input", required=True, help="Candidate Negative Sample CSV File Path")
    parser.add_argument("--output_kept", required=True, help="Save the final retained negative sample CSV")
    parser.add_argument("--output_removed", required=True, help="Save Deleted Sample CSV")
    parser.add_argument("--email", required=True, help="NCBI Entrez email address")
    parser.add_argument("--api_key", default=None, help="NCBI API Key")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent threads (recommended 2-4)")
    parser.add_argument("--sample", type=int, default=0, help="Test sample size (0=full size)")
    parser.add_argument("--log_file", default="pubmed_filter.log", help="Log file path")
    args = parser.parse_args()
    

    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)


    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} candidate pairs")
    

    if args.sample > 0:
        df = df.sample(min(args.sample, len(df)))
        logger.info(f"Test pattern: sampling {len(df)} pairs")


    tasks = [(row, args.email, args.api_key) for _, row in df.iterrows()]
    hit_indices = set()
    
    logger.info(f"Start PubMed filtering (number of threads: {args.workers}))")
    with ThreadPool(args.workers) as pool:
        results = list(tqdm(
            pool.imap(worker, tasks),
            total=len(df),
            desc="PubMed filter",
            dynamic_ncols=True
        ))
    

    for idx, hit in results:
        if hit:
            hit_indices.add(idx)
    

    kept_df = df.drop(index=hit_indices)
    removed_df = df.loc[list(hit_indices)]
    

    kept_df.to_csv(args.output_kept, index=False)
    removed_df.to_csv(args.output_removed, index=False)
    
    logger.info(
        f"Filter Complete | Total: {len(df)} | "
        f"Keep: {len(kept_df)} | Remove: {len(removed_df)}"
    )
    logger.info(f"Retain samples saved to: {args.output_kept}")
    logger.info(f"Remove samples saved to: {args.output_removed}")
    logger.info(f"Detailed log view: {args.log_file}")


if __name__ == "__main__":
    main()


