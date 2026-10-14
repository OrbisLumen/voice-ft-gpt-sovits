"""Normalize generated local WAV files into a final subdirectory."""
import argparse
from pathlib import Path
import sys

import tts_utils as utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    config = utils.read_json(utils.ROOT/'config/narration.json')
    for value in args.files:
        source=(utils.ROOT/value).resolve()
        utils.finalize(config,source,source.parent/'final'/source.name)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    main()
