import os
import subprocess
import time

if __name__ == '__main__':
    for port in range(8001, 8101):
        subprocess.Popen(f"python ./node_server.py {port}", shell=True)
    time.sleep(999999)