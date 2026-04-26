import os
import sys
import time
import pandas as pd
from datetime import datetime

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from bullrun.core.pipeline import DailyInferencePipeline
from bullrun.utils.logger import get_logger

logger = get_logger("FinalValidation")

def run_validation_cycles(num_cycles=3):
    print("\n" + "="*60)
    print("🚀 BULLRUN FINAL VALIDATION: MULTI-CYCLE SIGNAL AUDIT")
    print("="*60)
    
    pipeline = DailyInferencePipeline()
    
    all_results = []
    
    for i in range(num_cycles):
        print(f"\n🔄 CYCLE {i+1}/{num_cycles} STARTING...")
        
        # We use force=True to bypass market timing and idempotency
        results = pipeline.run(force=True)
        all_results.append(results)
        
        trades = results.get('trades_executed', 0)
        decisions = results.get('decisions', [])
        
        print(f"✅ CYCLE {i+1} COMPLETE. Trades Executed: {trades}")
        
        # Sample Audit for the first ticker
        if decisions:
            d = decisions[0]
            print(f"📊 Sample Decision ({d['symbol']}): Conf: {d['confidence']:.3f} | RL: {d['rl_suggested']:.2f} | Action: {d['action']}")
        
        if i < num_cycles - 1:
            print("⏳ Waiting 5 seconds before next cycle...")
            time.sleep(5)

    print("\n" + "="*60)
    print("📈 FINAL AGGREGATE RESULTS")
    print("="*60)
    
    total_trades = sum(r['trades_executed'] for r in all_results)
    print(f"Total Trades across {num_cycles} cycles: {total_trades}")
    
    # Check for variance in Meta Probabilities
    meta_probs = []
    for r in all_results:
        for d in r['decisions']:
            meta_probs.append(d['confidence'])
            
    if len(set(meta_probs)) > 1:
        print("✅ VARIANCE CHECK: Meta Probabilities are varying (Not constant).")
    else:
        print("⚠️ VARIANCE CHECK: Meta Probabilities are CONSTANT. Potential issue.")

    if total_trades > 0:
        print("✅ EXECUTION CHECK: BUY/SELL generated. System is active.")
    else:
        print("⚠️ EXECUTION CHECK: Zero trades generated. Check thresholds or market logic.")

    print("\n[Audit Complete]")

if __name__ == "__main__":
    run_validation_cycles(3)
