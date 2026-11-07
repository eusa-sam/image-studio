"""Prepare the pinned official SANA-Sprint bundle outside the model inventory."""
import json
from pathlib import Path

from .prepare_flux2_klein import main as prepare

SPEC = json.loads((Path(__file__).parent/'manifests/sana-sprint-1.6b.json').read_text(encoding='utf-8'))


def main():
    prepare(spec=SPEC,slug='sana-sprint-1.6b')


if __name__ == '__main__':main()
