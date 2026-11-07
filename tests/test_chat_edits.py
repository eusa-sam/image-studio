import asyncio
import json
import tempfile
import threading
import unittest
from types import SimpleNamespace
from uuid import uuid4

from backend.app.chat import ChatInput, ChatService, resolve_suggestion
from backend.app.storage import SessionStore


class EditTests(unittest.TestCase):
    def body(self, session='test'):
        return ChatInput(session=session, request_id=uuid4(), prompt_id=uuid4(), expected_revision=0,
            model='test', device='cpu', image_mode='none', message='옷 색만 빨강으로',
            positive='Subject:\n  (blue jacket:1.2), blue eyes\nLighting: soft  light',
            negative='blur, (bad hands:1.1)', settings={'model':'image'})

    def content(self, edits, **overrides):
        return json.dumps(dict(reply='색만 수정했습니다.', update=True, positive='', negative='', edits=edits) | overrides)

    def test_exact_edits_preserve_format_and_do_not_cascade(self):
        body=self.body()
        edits=[dict(field='positive',old='blue jacket',new='blue eyes'),
               dict(field='positive',old='blue eyes',new='green eyes')]
        result=resolve_suggestion(self.content(edits),body)
        self.assertEqual(result['positive'], 'Subject:\n  (blue eyes:1.2), green eyes\nLighting: soft  light')
        self.assertEqual(result['negative'],body.negative)
        self.assertNotIn('edits',result)
        duplicate=json.loads(self.content(edits));duplicate.update(positive=result['positive'],negative=result['negative'])
        self.assertEqual(resolve_suggestion(json.dumps(duplicate),body),result)

    def test_ordered_wire_names_preserve_legacy_edit_semantics(self):
        body=self.body()
        for operation,old,new in [('replace','blue jacket','red jacket'),
                                  ('append','','\nsoft light'),('prepend','','portrait, ')]:
            with self.subTest(operation=operation):
                legacy=dict(field='positive',op=operation,old=old,new=new)
                wire=dict(action=operation,field='positive',source=old,text=new)
                self.assertEqual(resolve_suggestion(self.content([wire]),body),
                                 resolve_suggestion(self.content([legacy]),body))
        ambiguous=dict(action='replace',op='append',field='positive',source='blue jacket',text='red jacket')
        with self.assertRaises(ValueError):resolve_suggestion(self.content([ambiguous]),body)

    def test_full_replacement_omissions_preserve_other_field_and_explicit_edit_can_clear(self):
        body=self.body()
        result=resolve_suggestion(self.content([],positive=body.positive.replace('blue jacket','red jacket')),body)
        self.assertEqual(result['positive'],body.positive.replace('blue jacket','red jacket'))
        self.assertEqual(result['negative'],body.negative)
        result=resolve_suggestion(self.content([],negative='new negative'),body)
        self.assertEqual(result['positive'],body.positive)
        self.assertEqual(result['negative'],'new negative')
        cleared=resolve_suggestion(self.content([dict(field='negative',old=body.negative,new='')]),body)
        self.assertEqual(cleared['negative'],'')
        self.assertEqual(cleared['positive'],body.positive)

    def test_literal_insertions_preserve_boundaries_and_allow_explicit_replacement(self):
        body=self.body().model_copy(update={'positive':'  scene.\n'})
        result=resolve_suggestion(self.content([
            dict(field='positive',op='append',old='',new=', soft light '),
            dict(field='positive',op='prepend',old='',new='[Subject]\n'),
            dict(field='positive',old='scene',new='garden')]),body)
        self.assertEqual(result['positive'],'[Subject]\n  garden.\n, soft light ')
        self.assertEqual(result['negative'],body.negative)
        for edits in ([dict(field='positive',op='append',old='.',new='x')],
                      [dict(field='positive',op='replace',old='',new='x')],
                      [dict(field='positive',op='append',old='',new='x')]*2,
                      [dict(field='positive',op='append',old='',new='x'*16000)]):
            with self.subTest(edits=edits),self.assertRaises(ValueError):resolve_suggestion(self.content(edits),body)
        empty=body.model_copy(update={'positive':''})
        self.assertEqual(resolve_suggestion(self.content([dict(field='positive',op='append',old='',new='scene.')]),empty)['positive'],'scene.')

    def test_invalid_patches_reject_without_mutating_input(self):
        body=self.body(); original=body.model_dump()
        cases=[ [('blue','red')], [('missing','red')], [('blue jacket','red'),('jacket','coat')],
                [('blue jacket','x'*16000)] ]
        for pairs in cases:
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                resolve_suggestion(self.content([dict(field='positive',old=a,new=b) for a,b in pairs]),body)
        data=json.loads(self.content([dict(field='positive',old='blue jacket',new='red jacket')]))
        data['positive']='whole rewrite'
        with self.assertRaises(ValueError):resolve_suggestion(json.dumps(data),body)
        unchanged=resolve_suggestion(self.content([dict(field='positive',old='blue jacket',new='red jacket')],update=False),body)
        self.assertEqual(unchanged['positive'],body.positive)
        self.assertEqual(unchanged['negative'],body.negative)
        self.assertFalse(unchanged['update'])
        self.assertEqual(body.model_dump(),original)
        body=body.model_copy(update={'positive':'aaa'})
        with self.assertRaises(ValueError):
            resolve_suggestion(self.content([dict(field='positive',old='aa',new='b')]),body)

    def test_sync_and_stream_store_resolved_prompts_and_reject_ambiguous(self):
        content=self.content([dict(field='positive',old='blue jacket',new='red jacket')])
        class Model:
            def local_model(self,name):return {'digest':'test'},{}
            def release(self,name):pass
            def request(self,*args,**kwargs):return {'done':True,'message':{'content':content}}
            async def stream_chat(self,request):
                yield {'message':{'content':content[:30]}}
                yield {'message':{'content':content[30:]},'done':True}
        with tempfile.TemporaryDirectory() as root, SessionStore(root) as store:
            session=store.create_session('edit')
            service=ChatService(store,SimpleNamespace(gpu=threading.Lock(),suspend_image_model=lambda:None),Model())
            request=service._request(self.body(session),self.body(session).model_dump(mode='json'),{})
            self.assertIn('copy the requested replacement verbatim',request['messages'][0]['content'])
            async def stream(body):return [event async for event in service.stream(body)]
            for streaming in (False,True):
                body=self.body(session)
                result=asyncio.run(stream(body))[-1]['turn'] if streaming else service.send(body)
                self.assertEqual(result['status'],'SUCCEEDED')
                self.assertEqual(result['positive'],body.positive.replace('blue jacket','red jacket'))
                self.assertEqual(result['negative'],body.negative)
                self.assertEqual(result['context']['positive'],body.positive)
            content=self.content([dict(field='positive',old='blue',new='red')])
            for streaming in (False,True):
                body=self.body(session)
                result=asyncio.run(stream(body))[-1]['turn'] if streaming else service.send(body)
                self.assertEqual(result['status'],'FAILED')
                self.assertNotIn('positive',result)
            self.assertEqual(store.prompts(session),[])


if __name__=='__main__':unittest.main()
