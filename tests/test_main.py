import unittest

from theseus.main import read_config


class TestMainModule(unittest.TestCase):

    def test_read_config_from_example_dir_with_json_ending(self):
        config, filename = read_config(is_example=True, filename='ghz_346.json')

        self.assertEqual(
            config['target_state'], ["000", "111", "222", "333"]
        )

    def test_read_config_from_example_dir_without_json_ending(self):
        config, filename = read_config(is_example=True, filename='ghz_346')

        self.assertEqual(
            config['target_state'], ["000", "111", "222", "333"]
        )
        self.assertEqual('ghz_346.json', filename.name)
