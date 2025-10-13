import sys
import asyncio
import logging
import time
import json
from pathlib import Path

import pandas as pd

# Add the project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic

from atom.llm_output_parsing.langchain_output_parser import LangchainOutputParser
from atom.models import AtomicFactsJudge, AtomicFactsPrompt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
    ]
)
logger = logging.getLogger(__name__)

print("🚀 Starting exhaustivity evaluation script...")
logger.info("Setting up API connections...")

# ==========================
# Global configuration vars
# ==========================
# Paths
INPUT_DATASET_PATH: Path = project_root / "datasets" / "nyt_news" / "subset_2020_nyt_COVID_evaluated_mistral.pkl"
OUTPUT_DATASET_PATH: Path = project_root / "datasets" / "nyt_news" / "subset_2020_nyt_COVID_evaluated_o3mini.pkl"

# Column names
GROUND_TRUTH_COL_NAME: str = "cumul_factoids_g_truth"
PREDICTED_COL_NAME: str = "cumul_factoids_o3mini"
EVALUATION_COL_NAME: str = f"{PREDICTED_COL_NAME}_evaluator"

# Batch processing configuration
BATCH_SIZE: int = 5  # Process 5 evaluations per batch (smaller due to complex LLM evaluation)
CHECKPOINT_FILE: Path = project_root / "datasets" / "nyt_news" / "exhaustivity_evaluation_checkpoint.json"

mistral_api_key = "###"
mistral_llm_model = ChatMistralAI(
    api_key = mistral_api_key,
    model="mistral-large-latest",
    temperature=0,
    max_retries=2,
)

mistral_embeddings_model = MistralAIEmbeddings(
    model="mistral-embed",
    api_key = mistral_api_key
)

openai_api_key = "###"

#gpt-4o-2024-11-20
#gpt-4.1-2025-04-14
#o3-mini-2025-01-31
#gpt-4-turbo-2024-04-09

openai_llm_model = ChatOpenAI(
    api_key = openai_api_key,
    model="gpt-4-turbo-2024-04-09",  # Better structured output support
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

claude_api_key = "###"

claude_llm_model = ChatAnthropic(
    api_key= claude_api_key,
    model="claude-sonnet-4-20250514",
    temperature=0,
    timeout=None,
    max_tokens=64000,
    max_retries=2,
)

openai_embeddings_model = OpenAIEmbeddings(
    api_key = openai_api_key ,
    model="text-embedding-3-large",
)

# Initialize LLM judge for evaluation
lg_llm_judge = LangchainOutputParser(
   llm_model=openai_llm_model,
   embeddings_model=openai_embeddings_model
)

logger.info("✅ LangchainOutputParser initialized successfully")

print("📊 Loading dataset...")
df_nyt = pd.read_pickle(INPUT_DATASET_PATH)
logger.info(f"📋 Loaded dataset with {len(df_nyt)} rows")

def load_checkpoint() -> dict:
    """Load checkpoint data if it exists, otherwise return empty checkpoint."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        logger.info(f"📂 Loaded checkpoint: {len(checkpoint.get('completed_batches', []))} batches completed")
        return checkpoint
    else:
        logger.info("📂 No checkpoint found, starting fresh")
        return {"completed_batches": [], "results": {}}

def save_checkpoint(checkpoint: dict):
    """Save current progress to checkpoint file."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"💾 Checkpoint saved: {len(checkpoint['completed_batches'])} batches completed")

async def evaluate_atomic_facts(ground_truth_factoids: list[list[str]], predicted_factoids: list[list[str]]) -> list[AtomicFactsJudge]:
    """Evaluate atomic facts for a batch of ground truth and predicted factoids."""
    logger.info(f"🔍 Starting atomic facts evaluation for batch of {len(ground_truth_factoids)} pairs...")
    
    try:
        stats = await lg_llm_judge.extract_information_as_json_for_context(
            output_data_structure=AtomicFactsJudge,
            contexts=[AtomicFactsPrompt.atomic_facts_system_query(ground_truth=ground_truth, predicted=predicted)
            for ground_truth, predicted in zip(ground_truth_factoids, predicted_factoids)],
            system_query=AtomicFactsPrompt.system_query.value,
        )
        
        logger.info(f"✅ Completed evaluation for {len(stats)} pairs")
        return stats
        
    except Exception as e:
        logger.error(f"❌ Error during atomic facts evaluation: {str(e)}")
        # Return empty results for failed evaluations
        return [AtomicFactsJudge(MATCH=0) for _ in range(len(ground_truth_factoids))]

async def evaluate_factoids_batch(ground_truth_batch: list[list[str]], predicted_batch: list[list[str]]) -> list[tuple[AtomicFactsJudge, int, int]]:
    """Evaluate a batch of factoids pairs. Returns list of (AtomicFactsJudge, n_predicted, n_gold) tuples."""
    logger.info(f"🔍 Starting factoids evaluation for batch of {len(ground_truth_batch)} pairs...")
    
    batch_results = await evaluate_atomic_facts(
        ground_truth_factoids=ground_truth_batch, 
        predicted_factoids=predicted_batch
    )
    
    # Combine results with counts for metrics calculation
    enhanced_results = []
    for i, result in enumerate(batch_results):
        n_predicted = len(predicted_batch[i]) if predicted_batch[i] else 0
        n_gold = len(ground_truth_batch[i]) if ground_truth_batch[i] else 0
        enhanced_results.append((result, n_predicted, n_gold))
    
    logger.info(f"✅ Batch evaluation completed: {len(enhanced_results)} evaluations")
    return enhanced_results

async def main():
    start_time = time.time()
    
    try:
        print("🎯 Starting main evaluation process...")
        logger.info("Beginning exhaustivity evaluation on NYT COVID data")
        
        # Load checkpoint
        checkpoint = load_checkpoint()
        
        # Filter for rows that have both ground truth and predicted data
        valid_mask = (df_nyt[GROUND_TRUTH_COL_NAME].notna()) & (df_nyt[PREDICTED_COL_NAME].notna())
        valid_indices = df_nyt[valid_mask].index.tolist()
        
        logger.info(f"📝 Processing {len(valid_indices)} rows with both ground truth and predicted data")
        
        if len(valid_indices) == 0:
            logger.warning("⚠️ No valid rows found for evaluation!")
            return

        # Create batches from valid indices
        batches = []
        for i in range(0, len(valid_indices), BATCH_SIZE):
            batch_indices = valid_indices[i:i + BATCH_SIZE]
            batches.append(batch_indices)
        
        logger.info(f"📦 Created {len(batches)} batches of size {BATCH_SIZE}")

        # Initialize the results column if not exists
        if EVALUATION_COL_NAME not in df_nyt.columns:
            df_nyt[EVALUATION_COL_NAME] = None

        # Load existing results from checkpoint
        for idx_str, result in checkpoint.get("results", {}).items():
            idx = int(idx_str)
            if idx in df_nyt.index:
                # Convert dict back to AtomicFactsJudge object
                if isinstance(result, dict):
                    atomic_facts_judge = AtomicFactsJudge(
                        MATCH=result.get("MATCH", 0)
                    )
                    df_nyt.at[idx, EVALUATION_COL_NAME] = atomic_facts_judge
                else:
                    df_nyt.at[idx, EVALUATION_COL_NAME] = result

        # Process batches
        for batch_idx, batch_indices in enumerate(batches):
            if batch_idx in checkpoint["completed_batches"]:
                logger.info(f"⏩ Skipping batch {batch_idx + 1}/{len(batches)} (already completed)")
                continue
                
            logger.info(f"🔄 Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_indices)} items)")
            
            # Prepare ground truth and predicted factoids for this batch
            batch_ground_truth = [df_nyt.loc[idx, GROUND_TRUTH_COL_NAME] for idx in batch_indices]
            batch_predicted = [df_nyt.loc[idx, PREDICTED_COL_NAME] for idx in batch_indices]
            
            # Handle None values by converting to empty lists
            batch_ground_truth = [gt if gt is not None else [] for gt in batch_ground_truth]
            batch_predicted = [pred if pred is not None else [] for pred in batch_predicted]
            
            # Evaluate factoids for this batch
            batch_results = await evaluate_factoids_batch(batch_ground_truth, batch_predicted)
            
            # Store results in dataframe and checkpoint
            for idx, (result, n_predicted, n_gold) in zip(batch_indices, batch_results):
                df_nyt.at[idx, EVALUATION_COL_NAME] = result
                # Store as dict in checkpoint for JSON serialization (store LLM result + counts for metrics calculation)
                checkpoint["results"][str(idx)] = {
                    "MATCH": result.MATCH,
                    "n_predicted": n_predicted,
                    "n_gold": n_gold,
                }
            
            # Mark batch as completed and save checkpoint
            checkpoint["completed_batches"].append(batch_idx)
            save_checkpoint(checkpoint)
            
            logger.info(f"✅ Batch {batch_idx + 1}/{len(batches)} completed and saved")

        # Save final results
        print(f"💾 Saving final results to: {OUTPUT_DATASET_PATH}")
        df_nyt.to_pickle(OUTPUT_DATASET_PATH)
        
        # Clean up checkpoint file
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            logger.info("🧹 Checkpoint file cleaned up")
        
        # Calculate and log aggregate metrics
        valid_evaluations = df_nyt[df_nyt[EVALUATION_COL_NAME].notna()]
        if len(valid_evaluations) > 0:
            # Calculate metrics for each evaluation and aggregate
            all_metrics = []
            total_match = 0
            total_predicted = 0
            total_gold = 0
            
            for idx, row in valid_evaluations.iterrows():
                eval_result = row[EVALUATION_COL_NAME]
                
                # Get the counts from checkpoint if available, otherwise estimate from current data
                idx_str = str(idx)
                if idx_str in checkpoint.get("results", {}):
                    checkpoint_data = checkpoint["results"][idx_str]
                    n_predicted = checkpoint_data.get("n_predicted", 0)
                    n_gold = checkpoint_data.get("n_gold", 0)
                else:
                    # Fallback: get counts from current data
                    predicted_facts = row.get(PREDICTED_COL_NAME, [])
                    ground_truth_facts = row.get(GROUND_TRUTH_COL_NAME, [])
                    n_predicted = len(predicted_facts) if predicted_facts else 0
                    n_gold = len(ground_truth_facts) if ground_truth_facts else 0
                
                # Calculate metrics for this evaluation
                metrics = eval_result.calculate_metrics(n_predicted=n_predicted, n_gold=n_gold)
                all_metrics.append(metrics)
                
                # Aggregate raw counts
                total_match += metrics["MATCH"]
                total_predicted += n_predicted
                total_gold += n_gold
            
            # Calculate overall aggregated metrics
            total_hall = total_predicted - total_match
            total_om = total_gold - total_match
            
            overall_precision = total_match / total_predicted if total_predicted > 0 else 0.0
            overall_recall = total_match / total_gold if total_gold > 0 else 0.0
            overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
            
            logger.info("📊 Aggregate Results:")
            logger.info(f"   MATCH: {total_match}, HALL: {total_hall}, OM: {total_om}")
            logger.info(f"   Total Predicted: {total_predicted}, Total Gold: {total_gold}")
            logger.info(f"   Precision: {overall_precision:.3f}")
            logger.info(f"   Recall: {overall_recall:.3f}")
            logger.info(f"   F1: {overall_f1:.3f}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"🎉 Evaluation completed successfully in {elapsed_time:.2f} seconds!")
        print(f"🎉 Exhaustivity evaluation completed successfully in {elapsed_time:.2f} seconds!")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ Error occurred after {elapsed_time:.2f} seconds: {str(e)}")
        print(f"❌ Error occurred: {str(e)}")
        print("💡 Progress has been saved. Re-run the script to resume from where it left off.")
        raise

if __name__ == "__main__":
    print("=" * 50)
    print("  EXHAUSTIVITY EVALUATION FOR NYT COVID DATA")
    print("=" * 50)
    asyncio.run(main())
