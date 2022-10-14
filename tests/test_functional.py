import os
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from pytheus.cli import run


class FunctionalTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('Running functional tests. This may take a while.')

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        os.chdir(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_input_without_json_ending_from_example_dir(self):
        runner = CliRunner()
        result = runner.invoke(run, ['--example', 'ghz_346'])
        assert result.exit_code == 0
        assert "{'|000000>': True, '|111000>': True, '|222000>': True, '|333000>': True}" in result.output
        assert os.path.exists('output/config_ghz_346/ghz_346/best.json')
        assert os.path.exists('output/config_ghz_346/ghz_346/summary.json')

    def test_input_with_json_ending_from_custom_dir(self):
        input_file = Path(__file__).parent / 'fixtures' / 'ghz_346.json'
        runner = CliRunner()
        result = runner.invoke(run, [str(input_file)])
        assert result.exit_code == 0
        assert "{'|000000>': True, '|111000>': True, '|222000>': True, '|333000>': True}" in result.output
        assert os.path.exists('output/ghz_346/ghz_346/best.json')
        assert os.path.exists('output/ghz_346/ghz_346/summary.json')

    def test_input_without_json_ending_from_custom_dir(self):
        input_file = Path(__file__).parent / 'fixtures' / 'ghz_346'
        runner = CliRunner()
        result = runner.invoke(run, [str(input_file)])
        assert result.exit_code == 0
        assert "{'|000000>': True, '|111000>': True, '|222000>': True, '|333000>': True}" in result.output
        assert os.path.exists('output/ghz_346/ghz_346/best.json')
        assert os.path.exists('output/ghz_346/ghz_346/summary.json')

    def test_non_existing_input_stops_with_error_message(self):
        runner = CliRunner()
        result = runner.invoke(run, ['i_dont_exist.json'])
        assert result.exit_code == 1
        assert 'ERROR' in result.output

    def test_bell_state(self):
        input_file = Path(__file__).parent / 'fixtures' / 'bell'
        runner = CliRunner()
        result = runner.invoke(run, [str(input_file)])
        assert 'finished with graph with 2 edges' in result.output

    def test_lossfunc_ent(self):
        runner = CliRunner()
        result = runner.invoke(run, ['--example', 'k2maximal4qubitsREAL'])
        assert result.exit_code == 0
        assert os.path.exists('output/config_k2maximal4qubitreal/try/best.json')
        assert os.path.exists('output/config_k2maximal4qubitreal/try/summary.json')

    def test_input_without_json_ending_from_example_director_removeConnections(self):
        runner = CliRunner()
        result = runner.invoke(run, ['--example', 'toffoli_post'])
        assert result.exit_code == 0

    def test_input_without_json_ending_from_example_director_startinggraph(self):
        runner = CliRunner()
        result = runner.invoke(run, ['--example', 'nbody3'])
        assert result.exit_code == 0

    @unittest.skip #does not exist anymore
    def test_input_with_json_ending_from_example1_director(self):
        runner = CliRunner()
        result = runner.invoke(run, ['--example', 'werner.json'])
        assert result.exit_code == 0