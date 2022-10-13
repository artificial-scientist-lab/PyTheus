import os
from pathlib import Path

package = 'theseus'

package_dir = Path.cwd().parent / package
assert os.path.exists(package_dir)


modules = filter(
    lambda f: os.path.isfile(package_dir / f) and f.endswith('.py') and not f == '__init__.py',
    os.listdir(package_dir)
)

template = """
Module `{}`
{}

.. automodule:: {}.{}
    :members:

"""

api_docs = 'API\n===\n\n'

for module in modules:
    module = module[:-3]
    api_docs += template.format(
        module, '-' * (len(package) + 9),
        package, module
    )

print(api_docs)