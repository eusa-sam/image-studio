"""Fast-forward a clean checkout; preserve local data and recover a failed sync."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone
from filelock import FileLock
from . import server


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True,encoding='utf-8').strip()


def sync_environment(root):
    subprocess.run(['uv','sync','--locked'],cwd=root,check=True)


def check_public_tree(root,revision,protected=()):
    for name in filter(None,git(root,'ls-tree','-rz','--name-only',revision).split('\0')):
        path=Path(name);first=path.parts[0].lower()
        reserved=any((root/path).resolve().is_relative_to(folder) for folder in protected)
        if reserved or first in ('data','models','outputs','docs','agents.md','.venv','.git') or (first.startswith('.env') and not first.endswith('.example')) or path.suffix.lower() in ('.key','.pem'):
            raise RuntimeError('Update contains a protected local path: '+name)


def update(root,runtime,data_root,port):
    if not shutil.which('uv') or not shutil.which('git'):raise RuntimeError('uv and Git are required')
    if git(root,'status','--porcelain'):raise RuntimeError('Save or commit local source changes before updating')
    before=git(root,'rev-parse','HEAD')
    upstream=git(root,'rev-parse','--abbrev-ref','--symbolic-full-name','@{upstream}')
    git(root,'fetch','--no-tags')
    target=git(root,'rev-parse',upstream)
    git(root,'merge-base','--is-ancestor',before,target)
    protected=[data_root.resolve()]
    for name in ('application','container'):
        settings=data_root/'settings'/f'{name}.json'
        if settings.exists():
            values=json.loads(settings.read_text(encoding='utf-8'))
            protected.extend(Path(values[key]).resolve() for key in ('model_path','output_path'))
    check_public_tree(root,before,protected);check_public_tree(root,target,protected)
    if git(root,'status','--porcelain'):raise RuntimeError('Source changed during update preparation')
    runtime.mkdir(parents=True,exist_ok=True)
    with FileLock(runtime/'update.lock',timeout=0):
        if git(root,'rev-parse','HEAD')!=before or git(root,'status','--porcelain'):
            raise RuntimeError('Source changed before the update lock was acquired')
        state=server.status(runtime)
        if state['state'] in ('starting','stopping'):raise RuntimeError('Wait for the server lifecycle operation to finish')
        running=state['state']=='running'
        port=state.get('port',port)
        record=dict(before=before,target=target,was_running=running,status='prepared')
        log=runtime/('update-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
        server.write(log,record)
        try:
            if running:server.stop(runtime)
            git(root,'merge','--ff-only',target)
            sync_environment(root)
            if running:server.start(runtime,data_root,port)
            record['status']='succeeded'
        except Exception as error:
            record.update(status='failed',error=str(error))
            try:
                # --keep refuses conflicting edits instead of discarding them.
                server.stop(runtime)
                git(root,'reset','--keep',before)
                sync_environment(root)
                if running:server.start(runtime,data_root,port)
                record['rollback']='succeeded'
            except Exception as recovery:
                record['rollback']=str(recovery)
            raise
        finally:
            server.write(log,record)
        return dict(record,log=str(log))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,default=server.ROOT/'data/runtime/server')
    parser.add_argument('--data-root',type=Path,default=server.ROOT/'data')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    try:print(json.dumps(update(server.ROOT,args.runtime.resolve(),args.data_root.resolve(),args.port),indent=2))
    except Exception as error:
        print('Update failed: '+str(error),file=sys.stderr);raise SystemExit(1)


if __name__=='__main__':main()
