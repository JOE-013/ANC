#!/bin/bash
# Run this on YOUR machine (not the sandbox) — needs real internet access.
# Downloads a small clean-speech set + a noise set for Day 1.

set -e
mkdir -p raw_data && cd raw_data

echo "== Downloading LibriSpeech dev-clean (~337MB, ~5.4 hours of clean speech) =="
wget -c https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz

echo "== Downloading MUSAN noise subset (~1.1GB total for full corpus; noise/ folder only needed) =="
wget -c https://www.openslr.org/resources/17/musan.tar.gz
tar -xzf musan.tar.gz musan/noise   # extracts only the noise folder

echo "Done. You now have:"
echo "  raw_data/LibriSpeech/dev-clean/   <- clean speech (organized by speaker/chapter)"
echo "  raw_data/musan/noise/             <- generic noise clips (free-sound, sound-bible, etc.)"
echo ""
echo "NEXT: manually source 3-5 defence-specific clips (gunshot, rotor, siren, artillery)"
echo "from freesound.org (filter by CC0 / CC-BY license) and place them in:"
echo "  raw_data/defence_noise/"
