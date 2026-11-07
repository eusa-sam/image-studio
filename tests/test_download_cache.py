import os
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import downloads

class DownloadCacheTests(unittest.TestCase):
    def test_preparation_downloads_and_xet_stay_in_project(self):
        expected=Path(__file__).resolve().parents[1]/'data/cache/huggingface'
        self.assertEqual(downloads.CACHE_ROOT,expected)
        self.assertEqual(Path(os.environ['HF_HOME']),expected)
        self.assertEqual(Path(os.environ['HF_XET_CACHE']),expected/'xet')
        self.assertEqual(Path(os.environ['HF_HUB_CACHE']),expected/'hub')
        with patch('huggingface_hub.hf_hub_download',return_value='prepared') as download:
            result=downloads.hf_hub_download('repository','weights',revision='a'*40,token=False)
        self.assertEqual(result,'prepared')
        download.assert_called_once_with('repository','weights',revision='a'*40,token=False,cache_dir=str(expected/'hub'))

    def test_uv_cache_is_project_relative(self):
        import tomllib
        root=Path(__file__).resolve().parents[1]
        config=tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))
        self.assertEqual(config['tool']['uv']['cache-dir'],'data/cache/uv')

    def test_preparation_scripts_still_import_as_direct_files(self):
        import subprocess,sys
        root=Path(__file__).resolve().parents[1]
        code="import sys,runpy;from pathlib import Path;sys.path.insert(0,str(Path('scripts').resolve()));[runpy.run_path(str(Path('scripts')/name),run_name='import_probe') for name in ['prepare_anima.py','prepare_anima_single.py','prepare_flux2_klein.py','prepare_ip_adapter.py','prepare_sd1.py','prepare_detectors.py']]"
        result=subprocess.run([sys.executable,'-c',code],cwd=root,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
