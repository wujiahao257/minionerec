"""无需训练依赖的目录迁移回归测试。运行：python -m unittest tests.test_repository_layout -v。"""

import ast
import importlib
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestRepositoryLayout(unittest.TestCase):
    """验证包结构、脚本入口及文档引用，避免文件移动后入口失效。

    Args:
        methodName (str): unittest 选择的测试方法名，由测试框架传入。
    """

    def test_python_syntax(self):
        """编译项目源码，发现语法错误而不导入模型依赖。

        Args:
            self (TestRepositoryLayout): 当前测试实例。
        """
        for directory in ('minionerec', 'rq', 'tests'):
            for path in (ROOT / directory).rglob('*.py'):
                with self.subTest(path=path.relative_to(ROOT)):
                    compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')

    def test_local_imports_resolve(self):
        """检查主代码包中的本地绝对导入和已移除的旧模块名。

        Args:
            self (TestRepositoryLayout): 当前测试实例。
        """
        old_names = {'data', 'utility', 'sasrec', 'SASRecModules_ori',
                     'LogitProcessor', 'minionerec_trainer', 'ts_rec_data'}
        paths = list((ROOT / 'minionerec').rglob('*.py')) + [ROOT / 'tests/test_datasets.py']
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if not isinstance(node, ast.ImportFrom) or node.level:
                    continue
                with self.subTest(path=path.relative_to(ROOT), module=node.module):
                    self.assertNotIn(node.module, old_names)
                    if node.module and node.module.startswith('minionerec.'):
                        target = ROOT.joinpath(*node.module.split('.'))
                        self.assertTrue(target.with_suffix('.py').is_file() or (target / '__init__.py').is_file())
                        if target.with_suffix('.py').is_file():
                            tree = ast.parse(target.with_suffix('.py').read_text(encoding='utf-8'))
                            definitions = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                            for alias in node.names:
                                if alias.name != '*':
                                    self.assertIn(alias.name, definitions)

    def test_packages_import_without_training_dependencies(self):
        """确认包初始化不加载模型、数据或训练依赖。

        Args:
            self (TestRepositoryLayout): 当前测试实例。
        """
        for path in (ROOT / 'minionerec').rglob('__init__.py'):
            module = '.'.join(path.parent.relative_to(ROOT).parts)
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_shell_module_targets(self):
        """检查启动脚本的模块入口、工作目录定位以及 TS 专用入口。

        Args:
            self (TestRepositoryLayout): 当前测试实例。
        """
        for path in (ROOT / 'scripts').glob('*.sh'):
            text = path.read_text(encoding='utf-8')
            with self.subTest(script=path.name):
                self.assertIn('BASH_SOURCE[0]', text)
                self.assertIn('cd -- "$PROJECT_ROOT" || exit 1', text)
                targets = re.findall(r'-m (minionerec[.\w]+)', text)
                if path.name == 'convert_dataset.sh':
                    targets += re.findall(r'PYTHON_MODULE="([.\w]+)"', text)
                self.assertTrue(targets)
                for module in targets:
                    self.assertTrue(ROOT.joinpath(*module.split('.')).with_suffix('.py').is_file(), module)
                self.assertNotRegex(text, r'(?:python|torchrun|accelerate)[^\n]*\./(?:sft|rl|evaluate|split|merge|calc)\.py')
        ts = (ROOT / 'scripts/ts_rec_sft.sh').read_text(encoding='utf-8')
        self.assertIn('-m minionerec.experiments.ts.sft', ts)
        self.assertIn('./ts_rec_data/${category}.description_keywords.json', ts)

    def test_documentation_links(self):
        """确认说明文档指向的本地文件在迁移后仍然存在。

        Args:
            self (TestRepositoryLayout): 当前测试实例。
        """
        for path in [ROOT / 'README.md', *(ROOT / 'docs').glob('*.md')]:
            for link in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
                if link.startswith(('https:', 'http:', '#', 'mailto:')):
                    continue
                target = link.split('#')[0]
                with self.subTest(document=path.name, target=target):
                    self.assertTrue((path.parent / target).exists(), target)

    def test_root_has_no_python_or_shell_entries(self):
        """防止根目录重新出现重复的代码或启动入口。

        Args:
            self (TestRepositoryLayout): 当前测试实例。
        """
        self.assertEqual(list(ROOT.glob('*.py')), [])
        self.assertEqual(list(ROOT.glob('*.sh')), [])


if __name__ == '__main__':
    unittest.main()
