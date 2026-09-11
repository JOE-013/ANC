import os
import glob

def main():
    root_dir = r'd:\SIH 2026\DTLN\datasets\musan'
    base_contents = os.listdir(root_dir)
    print("Base contents:", base_contents)

    musan_base = os.path.join(root_dir, 'musan') if 'musan' in base_contents else root_dir
    subfolders = [d for d in os.listdir(musan_base) if os.path.isdir(os.path.join(musan_base, d))]
    
    print("\n=== MUSAN Dataset Summary ===")
    print(f"Extracted Path: {musan_base}")
    print(f"Subfolders Found: {subfolders}")

    total_bytes = 0
    for r, d, files in os.walk(musan_base):
        for f in files:
            total_bytes += os.path.getsize(os.path.join(r, f))
    
    print(f"Total Size: {total_bytes / (1024**3):.2f} GB ({total_bytes} bytes)")
    
    print("\n=== Subfolder Audio File Counts (.wav) ===")
    for sf in sorted(subfolders):
        sf_path = os.path.join(musan_base, sf)
        wav_files = glob.glob(os.path.join(sf_path, '**', '*.wav'), recursive=True)
        print(f"  - {sf:<10}: {len(wav_files):>5} .wav files")

if __name__ == '__main__':
    main()
