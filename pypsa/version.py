"""Version information for PyPSA package.

Examples
--------
>>> pypsa.__version__ # doctest: +SKIP
'0.34.0.post1.dev44+gf5e415b6'
>>> pypsa.__version_semver__ # doctest: +SKIP
'0.34.0'
>>> pypsa.__version_short__ # doctest: +SKIP
'0.34'
>>> pypsa.__version_semver_tuple__ # doctest: +SKIP
(0, 34, 0)
>>> pypsa.__version_short_tuple__ # doctest: +SKIP
(0, 34)

"""

import logging
import re
from importlib.metadata import version

logger = logging.getLogger(__name__)


def check_pypsa_version(version_string: str) -> None:
    """Check if the installed PyPSA version was resolved correctly."""
    if version_string.startswith("0.0"):
        logger.warning(
            "The correct version of PyPSA could not be resolved. This is likely due to "
            "a local clone without pulling tags. Please run `git fetch --tags`."
        )


# e.g. "0.17.1" or "0.17.1.dev4+ga3890dc0" (if installed from git)
__version__ = version("pypsa")

# e.g. "0.17.0"
match = re.match(r"(\d+\.\d+(\.\d+)?)", __version__)
if not match:
    msg = f"Could not determine release_version of pypsa: {__version__}"
    raise ValueError(msg)

__version_semver__ = match.group(0)
__version_semver_tuple__ = tuple(map(int, __version_semver__.split(".")))
# e.g. "0.17"
match = re.match(r"(\d+\.\d+)", __version__)

if not match:
    msg = f"Could not determine release_version_short of pypsa: {__version__}"
    raise ValueError(msg)

__version_short__ = match.group(1)
__version_short_tuple__ = tuple(map(int, __version_short__.split(".")))

# Check pypsa version
check_pypsa_version(__version__)
