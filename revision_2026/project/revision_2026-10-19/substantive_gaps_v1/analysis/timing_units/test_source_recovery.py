"""Tests use constructed parser inputs, not clinical records."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import unittest
from recover_nested_source import literal_container
from audit_residence_fields import classify
from compare_native_city_values import helpers

class LiteralSafetyTests(unittest.TestCase):

    def test_python_container_not_repaired(self):
        value, status = literal_container("[{'field': 'value', 'clock': None}]")
        self.assertEqual(status, 'python_literal_container')
        self.assertEqual(value, [{'field': 'value', 'clock': None}])

    def test_calls_rejected(self):
        value, status = literal_container("[__import__('os').getcwd()]")
        self.assertIsNone(value)
        self.assertEqual(status, 'nonliteral_ast_node')

    def test_unquoted_strings_not_guessed(self):
        value, status = literal_container('[{field=value}]')
        self.assertIsNone(value)
        self.assertEqual(status, 'not_python_expression')

    def test_scalar_not_container(self):
        self.assertEqual(literal_container('42'), (None, 'not_literal_container_root'))

    def test_comprehension_rejected(self):
        value, _ = literal_container('[x for x in range(10)]')
        self.assertIsNone(value)

class ResidenceTests(unittest.TestCase):

    def test_exact_name_only(self):
        self.assertEqual(classify('City A', {'City A'}), ('exact_city_workbook_name', 'City A'))
        self.assertEqual(classify('City A District', {'City A'}), ('plain_text_not_city_name', None))

    def test_template_is_not_residence(self):
        self.assertEqual(classify('<select>City A</select>', {'City A'}), ('template_or_html', None))

    def test_pure_verified_source_link_functions(self):
        h = helpers()
        record = {h['EMR']: [{h['PERSONAL']: [{h['PID']: 'test-key'}]}, {h['STAY']: [{h['ADM']: '2020-01-01'}, {h['DIS']: '2020-01-03'}]}]}
        pid, key, _, flags = h['source_link'](record, h)
        self.assertEqual(pid, 'test-key')
        self.assertEqual(str(key[1]), '2020-01-01 00:00:00')
        self.assertFalse(any(flags.values()))
if __name__ == '__main__':
    unittest.main()
