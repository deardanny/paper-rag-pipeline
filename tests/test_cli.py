"""Test CLI input handling without importing API clients or making API calls."""
import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def cli_namespace():
    path = Path(__file__).resolve().parents[1] / 'app.py'
    nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)]
    context = {'os': os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), context)
    context['load_paper'] = Mock(return_value=['fictional chunk'])
    context['OpenAIEmbeddings'] = Mock()
    context['ChatOpenAI'] = Mock()
    context['RunnablePassthrough'] = Mock()
    context['Chroma'] = Mock()

    class Chain:
        invoke = Mock(return_value=SimpleNamespace(content='Fictional answer'))

        def __ror__(self, other):
            return self

        def __or__(self, other):
            return self

    chain = Chain()
    context['ChatPromptTemplate'] = SimpleNamespace(from_template=Mock(return_value=chain))
    return context, chain


class CLIInputTests(unittest.TestCase):
    def test_cancel_at_path_prompt_does_not_load_or_embed(self):
        for error in [EOFError, KeyboardInterrupt]:
            with self.subTest(error=error):
                context, _ = cli_namespace()
                with patch('builtins.input', side_effect=error), redirect_stdout(io.StringIO()):
                    context['main']()
                context['load_paper'].assert_not_called()
                context['OpenAIEmbeddings'].assert_not_called()

    def test_cancel_at_question_prompt_does_not_invoke_model(self):
        for error in [EOFError, KeyboardInterrupt]:
            with self.subTest(error=error):
                context, chain = cli_namespace()
                with patch('builtins.input', side_effect=['paper.pdf', error]), \
                     patch.object(os.path, 'isfile', return_value=True), redirect_stdout(io.StringIO()):
                    context['main']()
                chain.invoke.assert_not_called()

    def test_blank_question_then_answer_and_quit_still_work(self):
        context, chain = cli_namespace()
        output = io.StringIO()
        with patch('builtins.input', side_effect=[' paper.pdf ', '   ', ' What is it? ', ' QUIT ']), \
             patch.object(os.path, 'isfile', return_value=True), redirect_stdout(output):
            context['main']()
        context['load_paper'].assert_called_once_with('paper.pdf')
        chain.invoke.assert_called_once_with('What is it?')
        self.assertIn('Please enter a question.', output.getvalue())
        self.assertIn('Fictional answer', output.getvalue())


if __name__ == '__main__':
    unittest.main()
