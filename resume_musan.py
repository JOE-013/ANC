#!/usr/bin/env python3
import os
import subprocess
import tarfile

def main():
    url = 'https://www.openslr.org/resources/17/musan.tar.gz'
    tar_path = r'd:\SIH 2026\DTLN\datasets\musan.tar.gz'
    dest = r'd:\SIH 2026\DTLN\datasets\musan'

    print("Resuming MUSAN download from ~9.6 GB...")
    ret = subprocess.run(['curl.exe', '-C', '-', '-L', '-k', url, '-o', tar_path])
    print(f"Curl finished with return code: {ret.returncode}")

    if ret.returncode == 0:
        os.makedirs(dest, exist_ok=True)
        print("Extracting musan.tar.gz (~11 GB)...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=dest)
        if os.path.exists(tar_path):
            os.remove(tar_path)
        print("MUSAN dataset download and extraction completed successfully!")
    else:
        print("Curl download failed or was interrupted.")

if __name__ == '__main__':
    main()
