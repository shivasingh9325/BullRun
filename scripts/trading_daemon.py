import time
import subprocess
import sys
import os
from datetime import datetime

# Configuration
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 30
CHECK_INTERVAL_SEC = 300 # Check every 5 minutes

def run_daily_pipeline():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Launching Daily Pipeline...")
    try:
        # Run the manager tool
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(os.getcwd(), "src")
        result = subprocess.run([sys.executable, "manage.py", "daily"], env=env, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Error captured:\n{result.stderr}")
    except Exception as e:
        print(f"Critical Failure running pipeline: {e}")

def main():
    print("=== BullRun Trading Daemon Started ===")
    print(f"Target Execution: {MARKET_CLOSE_HOUR}:{MARKET_CLOSE_MINUTE} daily.")
    
    while True:
        now = datetime.now()
        
        # Simple Logic: If it's 4:30 PM and we haven't run today, execute.
        # Note: DailyInferencePipeline already has idempotency but we handle it here for logging.
        if now.hour == MARKET_CLOSE_HOUR and now.minute >= MARKET_CLOSE_MINUTE:
            # We add a sleep after running to ensure we don't trigger twice in the same minute
            run_daily_pipeline()
            print("Execution finished. Sleeping until 11:00 PM to reset cycle.")
            # Sleep for 6.5 hours (until 11 PM) to bypass the market close window
            time.sleep(23400) 
            
        time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main()
