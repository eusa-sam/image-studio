import os,socket,unittest
from unittest.mock import patch,Mock
from backend.app.ollama import Ollama,OllamaError,local_listener


class OllamaListenerTests(unittest.TestCase):
    @unittest.skipUnless(os.name=='nt','Windows listener table')
    def test_live_ipv4_and_ipv6_listeners_and_closed_port(self):
        for family,address in ((socket.AF_INET,'127.0.0.1'),(socket.AF_INET6,'::1')):
            with socket.socket(family) as server:
                server.bind((address,0));server.listen();port=server.getsockname()[1]
                self.assertIs(local_listener(port),True)
            self.assertIs(local_listener(port),False)

    def test_absent_service_keeps_manual_generation_and_chat_reports_error(self):
        with patch('backend.app.ollama.local_listener',return_value=False),patch('urllib.request.build_opener') as opener:
            Ollama().image_ready()
            with self.assertRaises(OllamaError):Ollama().models()
            opener.return_value.open.assert_not_called()

    def test_unknown_or_listening_still_checks_real_http_models(self):
        response=Mock();response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
        response.read.return_value=b'{"models":[{"name":"external","size_vram":100}]}'
        for state in (None,True):
            with patch('backend.app.ollama.local_listener',return_value=state),patch('urllib.request.build_opener') as opener:
                opener.return_value.open.return_value=response
                with self.assertRaises(OllamaError):Ollama().image_ready()
                opener.return_value.open.assert_called_once()

    @unittest.skipUnless(os.name=='nt','Windows listener table')
    def test_listener_query_failure_is_unknown_not_absent(self):
        with patch('ctypes.WinDLL',side_effect=OSError('unavailable')):self.assertIsNone(local_listener(11434))
