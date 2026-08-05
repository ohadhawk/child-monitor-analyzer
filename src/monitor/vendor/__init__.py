"""
Vendored third-party code.

Modules under this package are derived from external open-source projects
and are bundled directly into the repository rather than installed from
PyPI.  See ``LICENSE-third-party.txt`` in each sub-package for the original
licences and attribution.

Rationale (supply-chain security):
    Vendoring removes an installable dependency from the runtime process.
    Since Python offers no in-process isolation, every installed package can
    read this application's credentials and audio recordings.  For small,
    unmaintained packages the safest option is to copy the few functions we
    actually use, audit them once, and pin them in version control.
"""

from __future__ import annotations
