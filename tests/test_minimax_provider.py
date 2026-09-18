"""Unit and integration tests for MiniMax LLM provider in MiniOneRec."""

import json
import os
import re
import sys
import types
import unittest
from unittest.mock import patch, MagicMock

# Mock heavy dependencies that may not be installed in test environment
_mock_modules = {}
for mod_name in [
    'torch', 'transformers', 'gensim', 'accelerate', 'accelerate.utils',
]:
    if mod_name not in sys.modules:
        _mock_modules[mod_name] = sys.modules[mod_name] = MagicMock()

# Add project root so we can import rq/text2emb/utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rq', 'text2emb'))

import utils as text2emb_utils


class TestGetResBatchDispatch(unittest.TestCase):
    """验证文本 API 请求会按 provider 分发到正确实现。

    Args:
        本类未定义独立构造参数；构造行为继承父类。
    """

    @patch.object(text2emb_utils, 'get_openai_batch', return_value=["openai result"])
    def test_default_provider_is_openai(self, mock_openai):
        """验证未填写 provider 时使用 OpenAI 分支。

        Args:
            self (TestGetResBatchDispatch): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_openai (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        api_info = {"api_key_list": ["key"]}
        result = text2emb_utils.get_res_batch("model", ["prompt"], 100, api_info)
        mock_openai.assert_called_once()
        self.assertEqual(result, ["openai result"])

    @patch.object(text2emb_utils, 'get_deepseek_batch', return_value=["deepseek result"])
    def test_deepseek_provider(self, mock_ds):
        """验证DeepSeek 分发分支。

        Args:
            self (TestGetResBatchDispatch): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_ds (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        api_info = {"provider": "deepseek", "api_key_list": ["key"]}
        result = text2emb_utils.get_res_batch("model", ["prompt"], 100, api_info)
        mock_ds.assert_called_once()
        self.assertEqual(result, ["deepseek result"])

    @patch.object(text2emb_utils, 'get_minimax_batch', return_value=["minimax result"])
    def test_minimax_provider(self, mock_mm):
        """验证MiniMax 分发分支。

        Args:
            self (TestGetResBatchDispatch): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_mm (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        api_info = {"provider": "minimax", "api_key_list": ["key"]}
        result = text2emb_utils.get_res_batch("model", ["prompt"], 100, api_info)
        mock_mm.assert_called_once()
        self.assertEqual(result, ["minimax result"])

    @patch.object(text2emb_utils, 'get_openai_batch', return_value=["fallback"])
    def test_unknown_provider_falls_back_to_openai(self, mock_openai):
        """验证未知 provider 回退到 OpenAI。

        Args:
            self (TestGetResBatchDispatch): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_openai (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        api_info = {"provider": "unknown_provider", "api_key_list": ["key"]}
        result = text2emb_utils.get_res_batch("model", ["prompt"], 100, api_info)
        mock_openai.assert_called_once()
        self.assertEqual(result, ["fallback"])


class TestMiniMaxBatch(unittest.TestCase):
    """验证 MiniMax 批量请求的顺序、地址和空输入行为。

    Args:
        本类未定义独立构造参数；构造行为继承父类。
    """

    def _make_response(self, content, status_code=200):
        """构造具有状态码和 chat/completions JSON 结构的模拟 HTTP 响应。

        Args:
            self (TestMiniMaxBatch): 当前实例，由 Python 在调用实例方法时自动传入。
            content (str): 模拟 API 返回的 message.content 文本。
            status_code (int): 模拟 HTTP 响应状态码，例如 200、429 或 500。

        Returns:
            unittest.mock.MagicMock: 可供 requests.post 替身返回的响应对象。
        """
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        return resp

    @patch.object(text2emb_utils.requests, 'post')
    def test_single_prompt_success(self, mock_post):
        """验证单条批量请求返回正常文本。

        Args:
            self (TestMiniMaxBatch): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        mock_post.return_value = self._make_response("Hello world")
        api_info = {"provider": "minimax", "api_key_list": ["test-key"]}
        results = text2emb_utils.get_minimax_batch(
            "MiniMax-M2.7", ["test prompt"], 100, api_info
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "Hello world")

    @patch.object(text2emb_utils.requests, 'post')
    def test_multiple_prompts(self, mock_post):
        """验证多条请求结果顺序和条数。

        Args:
            self (TestMiniMaxBatch): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        mock_post.side_effect = [
            self._make_response("Response 1"),
            self._make_response("Response 2"),
            self._make_response("Response 3"),
        ]
        api_info = {"provider": "minimax", "api_key_list": ["test-key"]}
        results = text2emb_utils.get_minimax_batch(
            "MiniMax-M2.7", ["p1", "p2", "p3"], 100, api_info
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], "Response 1")
        self.assertEqual(results[1], "Response 2")
        self.assertEqual(results[2], "Response 3")

    @patch.object(text2emb_utils.requests, 'post')
    def test_custom_base_url(self, mock_post):
        """验证自定义 API 地址被实际用于请求。

        Args:
            self (TestMiniMaxBatch): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        mock_post.return_value = self._make_response("ok")
        api_info = {
            "provider": "minimax",
            "api_key_list": ["key"],
            "base_url": "https://custom.api.example.com/v1"
        }
        text2emb_utils.get_minimax_batch("MiniMax-M2.7", ["p"], 100, api_info)
        call_url = mock_post.call_args[0][0]
        self.assertTrue(call_url.startswith("https://custom.api.example.com/v1"))

    @patch.object(text2emb_utils.requests, 'post')
    def test_default_base_url(self, mock_post):
        """验证未提供地址时使用默认 API 地址。

        Args:
            self (TestMiniMaxBatch): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        mock_post.return_value = self._make_response("ok")
        api_info = {"provider": "minimax", "api_key_list": ["key"]}
        text2emb_utils.get_minimax_batch("MiniMax-M2.7", ["p"], 100, api_info)
        call_url = mock_post.call_args[0][0]
        self.assertEqual(call_url, "https://api.minimax.io/v1/chat/completions")

    @patch.object(text2emb_utils.requests, 'post')
    def test_empty_prompt_list(self, mock_post):
        """验证空 prompt 列表不发网络请求。

        Args:
            self (TestMiniMaxBatch): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        api_info = {"provider": "minimax", "api_key_list": ["key"]}
        results = text2emb_utils.get_minimax_batch("MiniMax-M2.7", [], 100, api_info)
        self.assertEqual(results, [])
        mock_post.assert_not_called()


class TestSingleMiniMaxRequest(unittest.TestCase):
    """验证单次 MiniMax 请求的字段、重试和输出清理行为。

    Args:
        本类未定义独立构造参数；构造行为继承父类。
    """

    def _make_response(self, content, status_code=200):
        """构造具有状态码和 chat/completions JSON 结构的模拟 HTTP 响应。

        Args:
            self (TestSingleMiniMaxRequest): 当前实例，由 Python 在调用实例方法时自动传入。
            content (str): 模拟 API 返回的 message.content 文本。
            status_code (int): 模拟 HTTP 响应状态码，例如 200、429 或 500。

        Returns:
            unittest.mock.MagicMock: 可供 requests.post 替身返回的响应对象。
        """
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        return resp

    @patch.object(text2emb_utils.requests, 'post')
    def test_successful_request(self, mock_post):
        """验证单次请求成功返回文本。

        Args:
            self (TestSingleMiniMaxRequest): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        mock_post.return_value = self._make_response("result text")
        api_info = {"api_key_list": ["test-key"]}
        result = text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "prompt", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        self.assertEqual(result, "result text")

    @patch.object(text2emb_utils.requests, 'post')
    def test_think_tag_stripping(self, mock_post):
        """验证清除单行 think 标签。

        Args:
            self (TestSingleMiniMaxRequest): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        content = "<think>internal reasoning here</think>The actual answer"
        mock_post.return_value = self._make_response(content)
        api_info = {"api_key_list": ["test-key"]}
        result = text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "prompt", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        self.assertEqual(result, "The actual answer")

    @patch.object(text2emb_utils.requests, 'post')
    def test_multiline_think_tag_stripping(self, mock_post):
        """验证清除跨行 think 标签。

        Args:
            self (TestSingleMiniMaxRequest): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        content = "<think>\nStep 1: think\nStep 2: reason\n</think>\nFinal answer here"
        mock_post.return_value = self._make_response(content)
        api_info = {"api_key_list": ["test-key"]}
        result = text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "prompt", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        self.assertEqual(result, "Final answer here")

    @patch.object(text2emb_utils.requests, 'post')
    def test_rate_limit_retry(self, mock_post):
        """验证429 后重试并成功恢复。

        Args:
            self (TestSingleMiniMaxRequest): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        success = self._make_response("ok")
        mock_post.side_effect = [rate_limited, success]
        api_info = {"api_key_list": ["test-key"]}
        result = text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "prompt", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        self.assertEqual(result, "ok")
        self.assertEqual(mock_post.call_count, 2)

    @patch.object(text2emb_utils.requests, 'post')
    def test_server_error_returns_empty(self, mock_post):
        """验证服务端错误重试耗尽返回空字符串。

        Args:
            self (TestSingleMiniMaxRequest): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        error_resp = MagicMock()
        error_resp.status_code = 500
        mock_post.return_value = error_resp
        api_info = {"api_key_list": ["test-key"]}
        result = text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "prompt", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        self.assertEqual(result, "")
        self.assertEqual(mock_post.call_count, 3)

    @patch.object(text2emb_utils.requests, 'post')
    def test_request_payload_format(self, mock_post):
        """验证请求模型、消息、长度、温度和 stream 字段。

        Args:
            self (TestSingleMiniMaxRequest): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        mock_post.return_value = self._make_response("ok")
        api_info = {"api_key_list": ["test-key"], "temperature": 0.5}
        text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "test prompt", 256, api_info,
            "https://api.minimax.io/v1", 0
        )
        payload = mock_post.call_args[1]['json']
        self.assertEqual(payload['model'], "MiniMax-M2.7")
        self.assertEqual(payload['messages'][0]['content'], "test prompt")
        self.assertEqual(payload['max_tokens'], 256)
        self.assertEqual(payload['temperature'], 0.5)
        self.assertFalse(payload['stream'])

    @patch.object(text2emb_utils.requests, 'post')
    def test_authorization_header(self, mock_post):
        """验证Bearer 授权头的构造。

        Args:
            self (TestSingleMiniMaxRequest): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        mock_post.return_value = self._make_response("ok")
        api_info = {"api_key_list": ["my-secret-key"]}
        text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        headers = mock_post.call_args[1]['headers']
        self.assertEqual(headers['Authorization'], "Bearer my-secret-key")

    @patch.object(text2emb_utils.requests, 'post')
    def test_exception_retries(self, mock_post):
        """验证连接异常后重试并恢复。

        Args:
            self (TestSingleMiniMaxRequest): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        mock_post.side_effect = [
            Exception("Connection error"),
            Exception("Timeout"),
            self._make_response("recovered")
        ]
        api_info = {"api_key_list": ["key"]}
        result = text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        self.assertEqual(result, "recovered")

    @patch.object(text2emb_utils.requests, 'post')
    def test_no_think_tag_passthrough(self, mock_post):
        """验证不含 think 标签的文本保持原样。

        Args:
            self (TestSingleMiniMaxRequest): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        mock_post.return_value = self._make_response("plain answer without thinking")
        api_info = {"api_key_list": ["key"]}
        result = text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        self.assertEqual(result, "plain answer without thinking")

    @patch.object(text2emb_utils.requests, 'post')
    def test_whitespace_stripping(self, mock_post):
        """验证回复首尾空白被清理。

        Args:
            self (TestSingleMiniMaxRequest): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        mock_post.return_value = self._make_response("  answer with spaces  ")
        api_info = {"api_key_list": ["key"]}
        result = text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        self.assertEqual(result, "answer with spaces")


class TestTemperatureClamping(unittest.TestCase):
    """验证 MiniMax 请求温度限制与边界值。

    Args:
        本类未定义独立构造参数；构造行为继承父类。
    """

    @patch.object(text2emb_utils.requests, 'post')
    def test_temperature_clamped_to_max_1(self, mock_post):
        """验证过高温度限制到 1。

        Args:
            self (TestTemperatureClamping): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "ok"}}]
            })
        )
        api_info = {"api_key_list": ["key"], "temperature": 1.5}
        text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        payload = mock_post.call_args[1]['json']
        self.assertLessEqual(payload['temperature'], 1.0)

    @patch.object(text2emb_utils.requests, 'post')
    def test_temperature_clamped_to_min_0(self, mock_post):
        """验证负温度限制到 0。

        Args:
            self (TestTemperatureClamping): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "ok"}}]
            })
        )
        api_info = {"api_key_list": ["key"], "temperature": -0.5}
        text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        payload = mock_post.call_args[1]['json']
        self.assertGreaterEqual(payload['temperature'], 0.0)

    @patch.object(text2emb_utils.requests, 'post')
    def test_default_temperature(self, mock_post):
        """验证默认温度为 0.4。

        Args:
            self (TestTemperatureClamping): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "ok"}}]
            })
        )
        api_info = {"api_key_list": ["key"]}
        text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        payload = mock_post.call_args[1]['json']
        self.assertEqual(payload['temperature'], 0.4)

    @patch.object(text2emb_utils.requests, 'post')
    def test_temperature_0_accepted(self, mock_post):
        """验证温度下界 0 被接受。

        Args:
            self (TestTemperatureClamping): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "ok"}}]
            })
        )
        api_info = {"api_key_list": ["key"], "temperature": 0.0}
        text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        payload = mock_post.call_args[1]['json']
        self.assertEqual(payload['temperature'], 0.0)

    @patch.object(text2emb_utils.requests, 'post')
    def test_temperature_1_accepted(self, mock_post):
        """验证温度上界 1 被接受。

        Args:
            self (TestTemperatureClamping): 当前实例，由 Python 在调用实例方法时自动传入。
            mock_post (unittest.mock.MagicMock): patch 装饰器注入的替代调用对象，用来设置响应并断言调用参数。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "choices": [{"message": {"content": "ok"}}]
            })
        )
        api_info = {"api_key_list": ["key"], "temperature": 1.0}
        text2emb_utils._single_minimax_request(
            "MiniMax-M2.7", "p", 100, api_info,
            "https://api.minimax.io/v1", 0
        )
        payload = mock_post.call_args[1]['json']
        self.assertEqual(payload['temperature'], 1.0)


class TestMiniMaxIntegration(unittest.TestCase):
    """在配置 API key 时执行真实 MiniMax 请求的集成测试。

    Args:
        本类未定义独立构造参数；构造行为继承父类。
    """

    def setUp(self):
        """读取 MiniMax API key；未配置时跳过真实 API 集成测试。

        Args:
            self (TestMiniMaxIntegration): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 设置当前测试的 api_key 或触发 skipTest。
        """
        self.api_key = os.environ.get("MINIMAX_API_KEY")
        if not self.api_key:
            self.skipTest("MINIMAX_API_KEY not set")

    def test_single_completion(self):
        """验证真实 API 单条请求。

        Args:
            self (TestMiniMaxIntegration): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        api_info = {
            "provider": "minimax",
            "api_key_list": [self.api_key],
        }
        results = text2emb_utils.get_res_batch(
            "MiniMax-M2.5", ["Say 'hello' in one word."], 10, api_info
        )
        self.assertEqual(len(results), 1)
        self.assertIn("hello", results[0].lower())

    def test_batch_completion(self):
        """验证真实 API 多条请求。

        Args:
            self (TestMiniMaxIntegration): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        api_info = {
            "provider": "minimax",
            "api_key_list": [self.api_key],
        }
        prompts = [
            "What is 1+1? Answer with just the number.",
            "What is 2+2? Answer with just the number.",
        ]
        results = text2emb_utils.get_res_batch(
            "MiniMax-M2.5", prompts, 64, api_info
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(len(results[0]) > 0)
        self.assertTrue(len(results[1]) > 0)

    def test_end_to_end_preference_prompt(self):
        """验证真实 API 对偏好分析模板生成回复。

        Args:
            self (TestMiniMaxIntegration): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        api_info = {
            "provider": "minimax",
            "api_key_list": [self.api_key],
        }
        prompt = text2emb_utils.intention_prompt.format(
            dataset_full_name="Electronics",
            item_title="Wireless Bluetooth Headphones",
            review="Great sound quality and comfortable fit. Battery lasts all day."
        )
        results = text2emb_utils.get_res_batch(
            "MiniMax-M2.5", [prompt], 256, api_info
        )
        self.assertEqual(len(results), 1)
        self.assertTrue(len(results[0]) > 0)
        # The response should contain preference analysis
        result_lower = results[0].lower()
        self.assertTrue(
            "preference" in result_lower or "characteristics" in result_lower or "item" in result_lower
        )


class TestTestGenerationConfigHasNoTopKTopP(unittest.TestCase):
    """静态检查训练器测试生成配置关闭 top_k/top_p 过滤。

    Args:
        本类未定义独立构造参数；构造行为继承父类。
    """

    def test_trainer_source_sets_top_k_and_top_p_none(self):
        """验证训练器测试生成配置显式关闭 top_k/top_p。

        Args:
            self (TestTestGenerationConfigHasNoTopKTopP): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        import ast, os
        src = os.path.join(os.path.dirname(__file__), '..', 'minionerec', 'training', 'trainer.py')
        with open(src) as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr == 'test_generation_config':
                        call = node.value
                        if isinstance(call, ast.Call):
                            kw_map = {kw.arg: kw.value for kw in call.keywords}
                            self.assertIn('top_k', kw_map, "test_generation_config must include top_k=None")
                            self.assertIn('top_p', kw_map, "test_generation_config must include top_p=None")
                            self.assertIsInstance(kw_map['top_k'], ast.Constant)
                            self.assertIsNone(kw_map['top_k'].value, "top_k must be None")
                            self.assertIsInstance(kw_map['top_p'], ast.Constant)
                            self.assertIsNone(kw_map['top_p'].value, "top_p must be None")
                            return
        self.fail("Could not find test_generation_config assignment in minionerec_trainer.py")


if __name__ == '__main__':
    unittest.main()
