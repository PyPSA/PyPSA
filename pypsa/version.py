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


def parse_version_tuple(version_str: str) -> tuple[int | str, ...]:
    """Parse a semantic version string into a tuple of integers and an optional suffix.

    The function assumes the version string follows a simplified semantic
    versioning scheme with an optional pre-release suffix (e.g. ``rc1``, ``a2``,
    ``b3``). It splits the version into its numeric components and, if present,
    appends the suffix as a string.

    Parameters
    ----------
    version_str : str
        A version string such as ``"1.0.0"``, ``"1.0.0rc1"``, or ``"2.1a2"``.

    Returns
    -------
    tuple[int | str, ...]
        A tuple of integers for the numeric parts, and optionally a string
        suffix as the last element.

    Examples
    --------
    >>> parse_version_tuple("1.0.0")
    (1, 0, 0)
    >>> parse_version_tuple("1.0.0rc1")
    (1, 0, 0, 'rc1')
    >>> parse_version_tuple("2.1a2")
    (2, 1, 'a2')

    """
    # Split into numeric core + optional suffix (rc1, a1, b2, etc.)
    parts = re.split(r"([a-z]+\d*)", version_str, maxsplit=1)

    # Convert the numeric part into ints
    numbers = tuple(map(int, parts[0].split(".")))

    # Add suffix back if present
    if len(parts) > 1 and parts[1]:
        return numbers + (parts[1],)
    return numbers


# e.g. "0.17.1" or "0.17.1.dev4+ga3890dc0" (if installed from git)
__version__ = version("pypsa")

# e.g. "0.17.0"
match = re.match(r"(\d+\.\d+(?:\.\d+)?(?:[a-z]+\d*)?)", __version__)
if not match:
    msg = f"Could not determine release_version of pypsa: {__version__}"
    raise ValueError(msg)

__version_semver__ = match.group(0)
__version_semver_tuple__ = parse_version_tuple(__version_semver__)
# e.g. "0.17"
match = re.match(r"(\d+\.\d+)", __version__)

if not match:
    msg = f"Could not determine release_version_short of pypsa: {__version__}"
    raise ValueError(msg)

__version_short__ = match.group(1)
__version_short_tuple__ = tuple(map(int, __version_short__.split(".")))

# Check pypsa version
check_pypsa_version(__version__)
