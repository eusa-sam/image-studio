"""Prepare the pinned Z-Image-Turbo bundle without exposing partial downloads."""
import json
from pathlib import Path

from .prepare_flux2_klein import main as prepare

SPEC = json.loads((Path(__file__).parent/'manifests/z-image-turbo.json').read_text(encoding='utf-8'))


def main():
    prepare(spec=SPEC,slug='z-image-turbo')


if __name__ == '__main__':main()
