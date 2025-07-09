import os
import re
import pytest

import re

def camel_to_snake(name):
    if not name:
        return ""

    prefix_match = re.match(r'^([a-z]+)(?=[A-Z])', name)
    prefix = ''
    if prefix_match:
        prefix = prefix_match.group(1)
        name = name[len(prefix):]

    parts = re.findall(r'[A-Z]+(?=[A-Z][a-z0-9])|[A-Z]?[a-z0-9]+|[A-Z]+', name)

    result = '_'.join(parts).lower()
    
    if prefix:
        return f"{prefix}_{result}"
    else:
        return result

def test_all_architectures_have_tests():
    architectures_dir = "remvae/playground/architectures"
    tests_dir = "tests/architectures"

    missing_tests = []

    for architecture_name in os.listdir(architectures_dir):
        architecture_path = os.path.join(architectures_dir, architecture_name)
        if os.path.isdir(architecture_path):
            expected_test_dir = os.path.join(tests_dir, camel_to_snake(architecture_name))
            print(expected_test_dir)

            if not os.path.isdir(expected_test_dir):
                missing_tests.append(f"No tests implemented for: {architecture_name}")
            else:
                py_files = [f for f in os.listdir(expected_test_dir) if f.endswith('.py')]
                if not py_files:
                    missing_tests.append(f"Tests dir '{expected_test_dir}' does not contain any test")

    if missing_tests:
        pytest.fail("\n".join(missing_tests))