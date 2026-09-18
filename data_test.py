import unittest
import tempfile
import os
import pandas as pd
import json
import sys
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import (
    SFTData, D3Dataset, EvalD3Dataset, EvalSidDataset,
    SidDataset, SidSFTDataset, SidItemFeatDataset, RLTitle2SidDataset,
    RLSid2TitleDataset, RLSidhis2TitleDataset, FusionSeqRecDataset,
    TitleHistory2SidSFTDataset, PreferenceSFTDataset, UserPreference2sidSFTDataset
)
class MockTokenizer:
    """按文本长度生成整数序列的测试替身，不执行真实分词。

    Args:
        本类未定义独立构造参数；构造行为继承父类。
    """
    def __init__(self):
        """初始化 MockTokenizer：按文本长度生成整数序列的测试替身，不执行真实分词。

        Args:
            self (MockTokenizer): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 完成实例初始化。
        """
        self.pad_token_id = 0
        self.eos_token_id = 3
        self.bos_token_id = 2
    
    def encode(self, text, bos=False, eos=False):
        # Simple mock encoding - just return list of integers based on text length
        """按文本长度生成模拟 token ID，按需加测试 BOS/EOS。

        Args:
            self (MockTokenizer): 当前实例，由 Python 在调用实例方法时自动传入。
            text (str): 需要模拟编码的文本；测试 tokenizer 仅根据长度生成编号。
            bos (bool): 是否在相应序列边界追加分词器定义的起始或结束 token。
            eos (bool): 是否在相应序列边界追加分词器定义的起始或结束 token。

        Returns:
            list[int]: 测试整数序列，不表达真实 token 语义。
        """
        tokens = list(range(10, 10 + min(len(text), 50)))  # Limit to 50 tokens max
        if bos:
            tokens = [self.bos_token_id] + tokens
        if eos:
            tokens = tokens + [self.eos_token_id]
        return tokens

def create_minimal_csv(file_path, data):
    """将测试字段字典转成 DataFrame 并写出最小交互 CSV。

    Args:
        file_path (str): 当前读写操作使用的文件或目录路径。
        data (dict[str, list]): 测试 CSV 的列名到列值列表映射。

    Returns:
        None: 创建测试输入文件。
    """
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

class TestDataModule(unittest.TestCase):
    """验证各类推荐 Dataset 能初始化并生成样本缓存。

    Args:
        本类未定义独立构造参数；构造行为继承父类。
    """
    @classmethod
    def setUpClass(cls):
        """创建共享临时目录、MockTokenizer、交互 CSV 和元数据/偏好 JSON 测试夹具。

        Args:
            cls (type[TestDataModule]): 当前测试类，由 classmethod 自动传入。

        Returns:
            None: 设置测试类共享属性。
        """
        cls.tokenizer = MockTokenizer()
        
        # Create temporary directory for test files
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = cls.temp_dir.name
        
        # Create sample CSV data
        cls.csv_data = {
            'history_item_title': ["['Item A', 'Item B']", "['Item C', 'Item D']"],
            'item_title': ['Item E', 'Item F'],
            'history_item_id': ["['1', '2']", "['3', '4']"],
            'item_id': ['5', '6'],
            'history_item_sid': ["['SID1', 'SID2']", "['SID3', 'SID4']"],
            'item_sid': ['SID5', 'SID6'],
            'user_id_original_str': ['user1', 'user2'],
            'e_token': ['[CTX_HOMEPAGE]', '[CTX_SEARCH]']
        }
        cls.csv_file = os.path.join(cls.temp_path, 'test_data.csv')
        create_minimal_csv(cls.csv_file, cls.csv_data)
        
        # Create sample item features JSON
        cls.item_features = {
            '5': {'title': 'Item E', 'description': 'Description of Item E', 'item_type': 'O'},
            '6': {'title': 'Item F', 'description': 'Description of Item F', 'item_type': 'I'}
        }
        cls.item_file = os.path.join(cls.temp_path, 'test.item.json')
        with open(cls.item_file, 'w') as f:
            json.dump(cls.item_features, f)
        
        # Create sample indices JSON
        cls.indices = {
            '5': ['SID5_1', 'SID5_2', 'SID5_3'],
            '6': ['SID6_1', 'SID6_2', 'SID6_3']
        }
        cls.index_file = os.path.join(cls.temp_path, 'test.index.json')
        with open(cls.index_file, 'w') as f:
            json.dump(cls.indices, f)
        
        # Create sample user preference JSON
        cls.user_preference_data = [
            {
                'user': 'user1',
                'user_preference': 'Likes action games',
                'context': {
                    'history_items': ['1', '2'],
                    'target_item': '5'
                },
                'split': 'train'
            },
            {
                'user': 'user2',
                'user_preference': 'Prefers strategy games',
                'context': {
                    'history_items': ['3', '4'],
                    'target_item': '6'
                },
                'split': 'train'
            }
        ]
        cls.preference_file = os.path.join(cls.temp_path, 'test_preference.json')
        with open(cls.preference_file, 'w') as f:
            json.dump(cls.user_preference_data, f)

    @classmethod
    def tearDownClass(cls):
        # Cleanup temporary directory
        """释放测试类创建的临时目录与数据文件。

        Args:
            cls (type[TestDataModule]): 当前测试类，由 classmethod 自动传入。

        Returns:
            None: 清理测试夹具。
        """
        cls.temp_dir.cleanup()

    def test_SFTData_initialization(self):
        """验证数据集 SFTData 的初始化、长度和缓存属性。

        Args:
            self (TestDataModule): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        dataset = SFTData(
            train_file=self.csv_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0,
            category="games"
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_D3Dataset_initialization(self):
        """验证数据集 D3Dataset 的初始化、长度和缓存属性。

        Args:
            self (TestDataModule): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        dataset = D3Dataset(
            train_file=self.csv_file,
            max_len=128,
            sample=1,
            seed=0,
            category="games"
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_EvalD3Dataset_initialization(self):
        """验证数据集 EvalD3Dataset 的初始化、长度和缓存属性。

        Args:
            self (TestDataModule): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        dataset = EvalD3Dataset(
            train_file=self.csv_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0,
            category="games"
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_SidDataset_initialization(self):
        """验证数据集 SidDataset 的初始化、长度和缓存属性。

        Args:
            self (TestDataModule): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        dataset = SidDataset(
            train_file=self.csv_file,
            max_len=128,
            sample=1,
            seed=0,
            category="games"
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_SidSFTDataset_initialization(self):
        """验证数据集 SidSFTDataset 的初始化、长度和缓存属性。

        Args:
            self (TestDataModule): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        dataset = SidSFTDataset(
            train_file=self.csv_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0,
            category="games"
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_SFTData_initialization(self):
        """验证数据集 SFTData 的初始化、长度和缓存属性。

        Args:
            self (TestDataModule): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        dataset = SFTData(
            train_file=self.csv_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0,
            category="games"
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_SidItemFeatDataset_initialization(self):
        """验证数据集 SidItemFeatDataset 的初始化、长度和缓存属性。

        Args:
            self (TestDataModule): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        dataset = SidItemFeatDataset(
            item_file=self.item_file,
            index_file=self.index_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=2,
            seed=0
        )
        self.assertGreaterEqual(len(dataset), 2)  # Should have at least 2 samples (sid2title and title2sid)
        self.assertTrue(hasattr(dataset, 'inputs'))
        
    def test_EvalSidDataset_initialization(self):
        """验证数据集 EvalSidDataset 的初始化、长度和缓存属性。

        Args:
            self (TestDataModule): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        dataset = EvalSidDataset(
            train_file=self.csv_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0,
            category="games"
        )

        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_RLTitle2SidDataset_initialization(self):
        """验证数据集 RLTitle2SidDataset 的初始化、长度和缓存属性。

        Args:
            self (TestDataModule): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        dataset = RLTitle2SidDataset(
            item_file=self.item_file,
            index_file=self.index_file,
            sample=2,
            seed=0
        )
        self.assertGreaterEqual(len(dataset), 2)  # Should have at least 2 samples
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_RLSid2TitleDataset_initialization(self):
        """验证数据集 RLSid2TitleDataset 的初始化、长度和缓存属性。

        Args:
            self (TestDataModule): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        dataset = RLSid2TitleDataset(
            item_file=self.item_file,
            index_file=self.index_file,
            sample=2,
            seed=0
        )
        self.assertGreaterEqual(len(dataset), 1)  # Should have at least 1 sample
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_RLSidhis2TitleDataset_initialization(self):
        """验证数据集 RLSidhis2TitleDataset 的初始化、长度和缓存属性。

        Args:
            self (TestDataModule): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        dataset = RLSidhis2TitleDataset(
            train_file=self.csv_file,
            item_file=self.item_file,
            index_file=self.index_file,
            sample=1,
            seed=0
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_FusionSeqRecDataset_initialization(self):
        """验证数据集 FusionSeqRecDataset 的初始化、长度和缓存属性。

        Args:
            self (TestDataModule): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        dataset = FusionSeqRecDataset(
            train_file=self.csv_file,
            item_file=self.item_file,
            index_file=self.index_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_TitleHistory2SidSFTDataset_initialization(self):
        """验证数据集 TitleHistory2SidSFTDataset 的初始化、长度和缓存属性。

        Args:
            self (TestDataModule): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        dataset = TitleHistory2SidSFTDataset(
            train_file=self.csv_file,
            item_file=self.item_file,
            index_file=self.index_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_PreferenceSFTDataset_initialization(self):
        """验证数据集 PreferenceSFTDataset 的初始化、长度和缓存属性。

        Args:
            self (TestDataModule): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        dataset = PreferenceSFTDataset(
            user_preference_file=self.preference_file,
            index_file=self.index_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

    def test_UserPreference2sidSFTDataset_initialization(self):
        """验证数据集 UserPreference2sidSFTDataset 的初始化、长度和缓存属性。

        Args:
            self (TestDataModule): 当前实例，由 Python 在调用实例方法时自动传入。

        Returns:
            None: 断言不成立时由 unittest 报告失败。
        """
        dataset = UserPreference2sidSFTDataset(
            user_preference_file=self.preference_file,
            index_file=self.index_file,
            tokenizer=self.tokenizer,
            max_len=128,
            sample=1,
            seed=0
        )
        self.assertEqual(len(dataset), 1)
        self.assertTrue(hasattr(dataset, 'inputs'))

if __name__ == '__main__':
    # Run the tests
    unittest.main()