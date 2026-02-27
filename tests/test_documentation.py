import unittest
import subprocess
import os

class TestDocumentation(unittest.TestCase):
    def test_generate_docs(self):
        # Path to the Sphinx documentation source directory
        docs_source_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')

        # Path to the output directory for the generated documentation
        docs_output_dir = os.path.join(docs_source_dir, '_build', 'html')

        # Command to generate the documentation
        command = ['sphinx-build', '--fail-on-warning', '-b',  'html', docs_source_dir, docs_output_dir]

        # Run the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True)

        # Check if the command was successful
        self.assertEqual(result.returncode, 0, f"Documentation generation failed: {result.stderr}")

        # Optionally, check if the output directory contains the expected files
        self.assertTrue(os.path.exists(docs_output_dir), "Output directory does not exist")
        self.assertTrue(os.path.isfile(os.path.join(docs_output_dir, 'index.html')),
                        "index.html not found in output directory")