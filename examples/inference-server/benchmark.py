import asyncio
import argparse
import time
import re
import statistics
import sys
from asyncio import create_subprocess_exec, subprocess

class BenchmarkStats:
    def __init__(self):
        self.rtfs = []
        self.latencies = []
        self.errors = 0
        self.success = 0
        self.start_time = 0
        self.end_time = 0

    def add_streaming_result(self, stdout):
        # Parse RTF and Processing Time from client-streaming.py output
        # "Processing Time: 0.17s"
        # "Real-Time Factor (RTF): 0.0847"
        try:
            rtf_match = re.search(r"Real-Time Factor \(RTF\): ([0-9.]+)", stdout)
            time_match = re.search(r"Processing Time: ([0-9.]+)s", stdout)
            
            if rtf_match and time_match:
                self.rtfs.append(float(rtf_match.group(1)))
                self.latencies.append(float(time_match.group(1)))
                self.success += 1
            else:
                self.errors += 1
        except:
            self.errors += 1

    def add_batch_result(self, duration, stdout, returncode):
        if returncode == 0:
            self.latencies.append(duration)
            self.success += 1
            # We don't have RTF easily for batch without file duration knowledge here, 
            # unless we parse it or pass it in. For now, latency is the main metric.
        else:
            self.errors += 1

    def report(self):
        total_time = self.end_time - self.start_time
        print("\n--- Benchmark Results ---")
        print(f"Total Requests:     {self.success + self.errors}")
        print(f"Successful:         {self.success}")
        print(f"Errors:             {self.errors}")
        print(f"Total Wall Time:    {total_time:.2f}s")
        print(f"Avg QPS:            {self.success / total_time:.2f}")

        if self.latencies:
            print("\n-- Latency (sec) --")
            print(f"Avg:  {statistics.mean(self.latencies):.3f}")
            print(f"P50:  {statistics.median(self.latencies):.3f}")
            print(f"P95:  {statistics.quantiles(self.latencies, n=20)[-1] if len(self.latencies) > 1 else self.latencies[0]:.3f}")

        if self.rtfs:
            print("\n-- RTF --")
            print(f"Avg:  {statistics.mean(self.rtfs):.4f}")
            print(f"P50:  {statistics.median(self.rtfs):.4f}")
            print(f"P95:  {statistics.quantiles(self.rtfs, n=20)[-1] if len(self.rtfs) > 1 else self.rtfs[0]:.4f}")

async def run_command(cmd):
    t0 = time.time()
    proc = await create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    t1 = time.time()
    return t1 - t0, stdout.decode(), stderr.decode(), proc.returncode

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["streaming", "batch"], required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--clients", type=int, default=1)
    parser.add_argument("--requests", type=int, default=1)
    args = parser.parse_args()

    print(f"Starting Benchmark: {args.clients} clients, {args.requests} requests each.")
    
    stats = BenchmarkStats()
    stats.start_time = time.time()

    # Semaphore to limit concurrency if needed, but here clients=concurrency
    sem = asyncio.Semaphore(args.clients)

    async def worker():
        async with sem:
            for _ in range(args.requests):
                if args.mode == "streaming":
                    cmd = ["python", "client-streaming.py", "-e", args.url, "-f", args.file]
                    dur, out, err, rc = await run_command(cmd)
                    if rc == 0:
                        stats.add_streaming_result(out)
                    else:
                        print(f"Error: {err}")
                        stats.errors += 1
                else:
                    # Batch uses curl
                    # curl -s -X POST url -F files=@file
                    cmd = ["curl", "-s", "-X", "POST", args.url, "-F", f"files=@{args.file}"]
                    dur, out, err, rc = await run_command(cmd)
                    if rc == 0 and "Error" not in out: # Simple check
                         stats.add_batch_result(dur, out, rc)
                    else:
                         stats.errors += 1

    tasks = [worker() for _ in range(args.clients)]
    await asyncio.gather(*tasks)
    
    stats.end_time = time.time()
    stats.report()

if __name__ == "__main__":
    asyncio.run(main())
