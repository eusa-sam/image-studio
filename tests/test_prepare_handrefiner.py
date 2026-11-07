import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from safetensors.numpy import save_file
from scripts import prepare_handrefiner

class HandRefinerPreparationTests(unittest.TestCase):
    def test_pinned_files_reuse_and_corruption_never_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'scripts/manifests').mkdir(parents=True)
            target=root/'models/sources/handrefiner/revision';target.mkdir(parents=True)
            readme=target/'README.md';readme.write_text('license information',encoding='utf8')
            weights=target/'control_sd15_inpaint_depth_hand_fp16.safetensors'
            save_file({'control.weight':np.zeros((2,3),dtype=np.float16)},weights)
            files=[dict(path=p.name,size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (readme,weights)]
            spec=dict(repository='example/test',revision='revision',license='apache-2.0',files=files)
            (root/'scripts/manifests/handrefiner-controlnet.json').write_text(json.dumps(spec),encoding='utf8')
            with patch.object(prepare_handrefiner,'__file__',str(root/'scripts/prepare_handrefiner.py')),patch.object(prepare_handrefiner,'hf_hub_download') as download:
                prepare_handrefiner.main();download.assert_not_called()
                report=json.loads((target/'preparation.json').read_text(encoding='utf8'))
                self.assertEqual(report['shapes'],{'control.weight':[2,3]})
                self.assertEqual(report['status'],'SOURCE_PREPARED_NOT_RUNTIME_SUPPORTED')
                weights.write_bytes(b'broken original')
                with self.assertRaises(ValueError):prepare_handrefiner.main()
                download.assert_not_called();self.assertEqual(weights.read_bytes(),b'broken original')
