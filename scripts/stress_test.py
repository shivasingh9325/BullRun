import asyncio
import httpx
import time

API_URL = "http://localhost:8000/api/v1/predict"

async def call_predict(i):
    print(f"[Run {i}] Calling /predict...")
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(API_URL)
            duration = time.time() - start
            if response.status_code == 200:
                print(f"[Run {i}] Success! Duration: {duration:.2f}s")
                # print(response.json())
            else:
                print(f"[Run {i}] Failed! Status: {response.status_code}")
    except Exception as e:
        print(f"[Run {i}] Network Error: {e}")

async def main():
    print("🚀 STARTING BULLRUN API STRESS TEST (Concurrency: 5)")
    start_total = time.time()
    
    # Fire 5 requests concurrently
    tasks = [call_predict(i) for i in range(5)]
    await asyncio.gather(*tasks)
    
    print(f"\n✅ STRESS TEST COMPLETE. Total Duration: {time.time() - start_total:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
