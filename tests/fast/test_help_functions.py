import unittest

from theseus.fancy_classes import State
from theseus.help_functions import readableState


class TestHelpFunctionsModule(unittest.TestCase):

    def test_readableState(self):
        term_list = ['000000', '111000', '222000', '333000']
        target_state = State(term_list, imaginary=False)
        expected = {'|000000>': True, '|111000>': True, '|222000>': True, '|333000>': True}
        self.assertEqual(expected, readableState(target_state))