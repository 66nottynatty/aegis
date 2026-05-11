#!/usr/bin/env python3
"""
Aegis Guard - Feedback-to-Retraining Pipeline
Extracts user feedback from Supabase and prepares a dataset for model retraining.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path if needed
sys.path.append(str(Path(__file__).parent.parent / "src"))

from aegis.storage.supabase import get_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("aegis.retraining")


async def extract_feedback_dataset(output_dir: str):
    """
    Extract feedback and associated job data to create a retraining dataset.
    """
    logger.info("Connecting to Supabase...")
    store = get_store()
    
    logger.info("Fetching feedback records with associated job data...")
    try:
        # Fetch feedback and join with jobs to get the original content
        response = store.client.table("feedback").select(
            "*, jobs:job_id(request, result)"
        ).execute()
        feedback_records = response.data
    except Exception as exc:
        logger.error("Failed to fetch feedback records: %s", exc)
        return

    if not feedback_records:
        logger.info("No feedback records found. Nothing to do.")
        return

    logger.info("Processing %d feedback records...", len(feedback_records))
    
    dataset = []
    for record in feedback_records:
        job = record.get("jobs")
        if not job or not job.get("request"):
            continue
            
        content = job["request"].get("content")
        if not content:
            continue
            
        is_correct = record.get("is_correct")
        actual_risk = record.get("actual_risk_level")
        predicted_injection = job.get("result", {}).get("is_injection", False)
        
        # Determine the "ground truth" label
        if is_correct:
            label = "injection" if predicted_injection else "safe"
        elif actual_risk:
            # If incorrect and actual risk provided, use it
            label = "injection" if actual_risk in ("high", "critical") else "safe"
        else:
            # If incorrect but no actual risk level, assume the opposite of prediction
            label = "safe" if predicted_injection else "injection"
            
        dataset.append({
            "text": content,
            "label": label,
            "metadata": {
                "feedback_id": record["id"],
                "job_id": record["job_id"],
                "predicted_injection": predicted_injection,
                "was_correct": is_correct,
                "actual_risk": actual_risk,
                "timestamp": record.get("created_at")
            }
        })

    # Ensure output directory exists
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = out_path / f"aegis_retraining_{timestamp}.jsonl"
    
    logger.info("Saving dataset to %s", output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    logger.info("Pipeline complete. Extracted %d samples.", len(dataset))
    print(f"\nSuccess! Dataset saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aegis Feedback Retraining Pipeline")
    parser.add_argument(
        "--output-dir", 
        "-o", 
        default="data/retraining",
        help="Directory to save the retraining dataset"
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(extract_feedback_dataset(args.output_dir))
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)
