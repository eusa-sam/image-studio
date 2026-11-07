import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import prepare_ip_adapter as prepare
from scripts.prepare_sd1 import checksum


class PrepareIPAdapterTests(unittest.TestCase):
    def test_pinned_preparation_reuse_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.write_bytes(b'weights')
            rows=[('remote','configs/ip_adapter/test.safetensors',checksum(source))]
            with patch.object(prepare,'ROOT',root),patch.object(prepare,'FILES',rows),patch.object(prepare,'hf_hub_download',return_value=str(source)) as download:
                prepare.main();prepare.main()
                download.assert_called_once_with(prepare.REPO,'remote',revision=prepare.REVISION,token=False)
                target=root/'models/configs/ip_adapter/test.safetensors';target.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'Existing preparation'):prepare.main()
                self.assertEqual(target.read_bytes(),b'changed')

    def test_wrong_download_is_not_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.write_bytes(b'wrong')
            with patch.object(prepare,'ROOT',root),patch.object(prepare,'FILES',[('remote','adapter/model.safetensors','0'*64)]),patch.object(prepare,'hf_hub_download',return_value=str(source)):
                with self.assertRaisesRegex(ValueError,'hash mismatch'):prepare.main()
                self.assertFalse((root/'models/adapter/model.safetensors').exists())
