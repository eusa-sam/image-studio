"""Prepare the pinned LongCat edit bundle outside the selectable inventory."""
import json
from pathlib import Path

from .prepare_flux2_klein import main as prepare

SPEC = json.loads((Path(__file__).parent/'manifests/longcat-image-edit-turbo.json').read_text(encoding='utf-8'))


def main():
    prepare(spec=SPEC,slug='longcat-image-edit-turbo')


if __name__ == '__main__':main()
