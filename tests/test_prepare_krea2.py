import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import prepare_krea2


class PrepareKreaTests(unittest.TestCase):
    def test_pinned_download_reuse_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(prepare_krea2,'ROOT',Path(directory)):
            calls=[]
            def download(url,timeout):calls.append(url);return io.BytesIO(b'{"test":true}')
            with patch.object(prepare_krea2.urllib.request,'urlopen',side_effect=download):
                prepare_krea2.main();prepare_krea2.main()
            self.assertEqual(len(calls),4)
            for url,source in zip(calls,prepare_krea2.SOURCES):self.assertIn('/'+source[1]+'/',url)
            target=Path(directory)/'vae/config.json';target.write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed'):prepare_krea2.main()
            self.assertEqual(target.read_text(),'{}')

    def test_partial_preparation_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(prepare_krea2,'ROOT',Path(directory)):
            path=Path(directory)/'text_encoder/config.json';path.parent.mkdir();path.write_text('{"keep":true}')
            with patch.object(prepare_krea2.urllib.request,'urlopen',return_value=io.BytesIO(b'{}')):
                with self.assertRaisesRegex(ValueError,'differs'):prepare_krea2.main()
            self.assertFalse((Path(directory)/'manifest.json').exists())
            self.assertEqual(path.read_text(),'{"keep":true}')
