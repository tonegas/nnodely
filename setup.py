from setuptools import setup, find_packages
import os

# import re
# def update_version_in_init(version_file):
#     with open(version_file, 'r') as f:
#         content = f.read()
#     new_version = os.environ.get('VERSION', '1.0.0')
#     content_new = re.sub(r'__version__ = ["\'].*["\']', f'__version__ = "{new_version}"', content)
#     with open(version_file, 'w') as f:
#         f.write(content_new)


def read_version():
    version_file = os.path.join(os.path.dirname(__file__), "nnodely", "__init__.py")
    with open(version_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name="nnodely",
    version=read_version(),
    packages=find_packages(exclude=["docs*", "tests*", "imgs*"]),
    include_package_data=True,
)
