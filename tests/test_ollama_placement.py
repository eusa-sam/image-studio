import asyncio
import json
import unittest
from unittest.mock import patch
import httpx
from backend.app.ollama import Ollama


class PlacementTests(unittest.TestCase):
    def test_stream_reports_actual_placement_once_without_changing_request(self):
        async def run(vram,options,broken=False):
            calls=[]
            def handle(request):
                calls.append(request.url.path)
                if request.url.path.endswith('/ps'):
                    if broken:return httpx.Response(503)
                    return httpx.Response(200,json={'models':[{'name':'test','size_vram':vram}]})
                self.assertEqual(json.loads(request.content)['options'],options)
                return httpx.Response(200,text='\n'.join(json.dumps(row) for row in (
                    {'message':{'thinking':'first'}},{'message':{'content':'answer'}},{'done':True})))
            client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
            with patch('httpx.AsyncClient',return_value=client):
                chunks=[row async for row in Ollama().stream_chat({'model':'test','options':options})]
            self.assertEqual(chunks[1],{'message':{'content':'answer'}})
            expected=None if broken or options.get('num_gpu')==0 else vram
            self.assertEqual(chunks[0].get('_gpu_bytes'),expected)
            self.assertEqual(calls.count('/api/ps'),0 if options.get('num_gpu')==0 else 1)
        for vram,options,broken in ((0,{},False),(123,{},False),(0,{'num_gpu':0},False),(0,{},True)):
            with self.subTest(vram=vram,options=options,broken=broken):asyncio.run(run(vram,options,broken))
