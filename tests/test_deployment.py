import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.app.main import create_app
from backend.app import models
from backend.app.app_settings import AppSettings
from scripts import update


class DeploymentTests(unittest.TestCase):
    def repository(self,root):
        upstream=root/'upstream';upstream.mkdir()
        update.git(upstream,'init','-b','main')
        (upstream/'.gitignore').write_text('data/\nmodels/\noutputs/\n.env\n.venv/\ncustom-models/\ncustom-output/\n')
        (upstream/'version.txt').write_text('old')
        self.commit(upstream)
        clone=root/'checkout';subprocess.run(['git','clone',str(upstream),str(clone)],check=True,capture_output=True)
        return upstream,clone

    def commit(self,root):
        update.git(root,'add','.')
        update.git(root,'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','fixture')

    def test_update_preserves_data_and_failed_sync_restores_source(self):
        for failing in (False,True):
            with self.subTest(failing=failing),tempfile.TemporaryDirectory() as directory:
                upstream,root=self.repository(Path(directory))
                data=root/'data';runtime=data/'runtime/server'
                files={'models/weight.safetensors':b'model','outputs/image.png':b'image',
                       'data/sessions/example/message.txt':b'keep','custom-models/weight':b'custom',
                       'custom-output/image.png':b'custom output','.env':b'private setting'}
                for name,content in files.items():
                    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
                settings=data/'settings/application.json';settings.parent.mkdir()
                settings.write_text(json.dumps(dict(model_path=str(root/'custom-models'),output_path=str(root/'custom-output'))))
                files[str(settings.relative_to(root))]=settings.read_bytes()
                before=update.git(root,'rev-parse','HEAD')
                (upstream/'version.txt').write_text('new');self.commit(upstream)
                effects=[RuntimeError('sync failed'),None] if failing else [None]
                with patch.object(update.server,'status',return_value={'state':'running'}),patch.object(update.server,'stop') as stop,patch.object(update.server,'start') as start,patch.object(update,'sync_environment',side_effect=effects) as sync:
                    if failing:
                        with self.assertRaisesRegex(RuntimeError,'sync failed'):update.update(root,runtime,data,8765)
                        self.assertEqual(update.git(root,'rev-parse','HEAD'),before)
                    else:self.assertEqual(update.update(root,runtime,data,8765)['status'],'succeeded')
                    stop.assert_called();start.assert_called_once_with(runtime,data,8765)
                    self.assertEqual(sync.call_count,2 if failing else 1)
                for name,content in files.items():self.assertEqual((root/name).read_bytes(),content,name)
                log=json.loads(next(runtime.glob('update-*.json')).read_text())
                self.assertEqual(log.get('rollback') if failing else log['status'],'succeeded')

    def test_dirty_checkout_and_incoming_protected_paths_stop_before_shutdown(self):
        for path in ('models/incoming','custom-models/incoming','docs/incoming',None):
            with self.subTest(path=path),tempfile.TemporaryDirectory() as directory:
                upstream,root=self.repository(Path(directory));data=root/'data';data.mkdir()
                (data/'settings').mkdir();(data/'settings/application.json').write_text(json.dumps(dict(model_path=str(root/'custom-models'),output_path=str(root/'outputs'))))
                if path:
                    target=upstream/path;target.parent.mkdir();target.write_text('must reject')
                    update.git(upstream,'add','-f',path);self.commit(upstream)
                else:(root/'version.txt').write_text('local edits')
                with patch.object(update.server,'stop') as stop,patch.object(update,'sync_environment') as sync:
                    with self.assertRaises(RuntimeError):update.update(root,data/'runtime/server',data,8765)
                    stop.assert_not_called();sync.assert_not_called()

    def test_container_settings_preserve_windows_settings_and_shared_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';(data/'settings').mkdir(parents=True)
            native=data/'settings/application.json'
            native.write_bytes(b'{"model_path":"D:/existing/models","output_path":"D:/existing/outputs"}')
            original=native.read_bytes();model=root/'models';model.mkdir()
            (model/'preserved.bin').write_bytes(b'original model')
            with patch.dict(os.environ,{'IMAGE_STUDIO_CONTAINER':'1'}),patch.object(models,'MODEL_ROOT',model):
                with TestClient(create_app(data),base_url='http://127.0.0.1:8765') as client:
                    settings=client.get('/api/settings').json()
                    self.assertEqual(settings['ollama_url'],'http://host.docker.internal:11434')
                    session=client.post('/api/sessions',json={'title':'Container workspace'}).json()['ref']
                    settings['output_path']=str(root/'outputs')
                    response=client.post('/api/settings',json=settings);self.assertEqual(response.status_code,200,response.text)
                with TestClient(create_app(data),base_url='http://127.0.0.1:8765') as client:
                    self.assertEqual(client.get('/api/settings').json()['output_path'],str(root/'outputs'))
                    self.assertIn(session,[row['ref'] for row in client.get('/api/sessions').json()])
            self.assertEqual(native.read_bytes(),original)
            self.assertEqual((model/'preserved.bin').read_bytes(),b'original model')
            self.assertTrue((data/'settings/container.json').is_file())
            with patch.dict(os.environ,{'IMAGE_STUDIO_CONTAINER':'0'}):
                with self.assertRaises(ValueError):AppSettings(model_path=str(model),output_path=str(root/'outputs'),ollama_url='http://host.docker.internal:11434')
