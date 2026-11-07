from html.parser import HTMLParser
from pathlib import Path
import unittest


class Markup(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.stack, self.nodes = [], {}
        self.feed(source)
        assert not self.stack, self.stack

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        identifier = attributes.get('id')
        if identifier:
            assert identifier not in self.nodes, identifier
            self.nodes[identifier] = (attributes, list(self.stack))
        if tag not in {'meta', 'link', 'input', 'img', 'br', 'hr'}:
            self.stack.append((tag, identifier))

    def handle_endtag(self, tag):
        assert self.stack and self.stack[-1][0] == tag, (tag, self.stack)
        self.stack.pop()


class WebLayoutTests(unittest.TestCase):
    def test_tabs_keep_editor_controls_and_results_in_separate_panels(self):
        nodes = Markup(Path('backend/app/web/index.html').read_text(encoding='utf-8')).nodes
        for identifier in ('positive', 'negative', 'model', 'clip', 'clip2', 'flow',
                           'width', 'height', 'seed', 'detail_steps', 'lora-slots', 'save', 'generate'):
            ancestors = nodes[identifier][1]
            self.assertIn(('form', 'editor'), ancestors)
            self.assertIn(('div', 'studio-create'), ancestors)
        self.assertIn(('div', 'studio-results'), nodes['jobs'][1])
        self.assertNotIn(('div', 'studio-create'), nodes['jobs'][1])
        self.assertNotIn(('div', 'studio-create'), nodes['chat-form'][1])
        for name in ('create', 'results'):
            self.assertEqual(nodes['tab-' + name][0]['aria-controls'], 'studio-' + name)
            self.assertEqual(nodes['studio-' + name][0]['aria-labelledby'], 'tab-' + name)
        self.assertEqual(nodes['chat-attach'][0]['type'], 'button')
        self.assertIn(('form', 'chat-form'), nodes['chat-send'][1])
