"""Loads components module."""

from __future__ import annotations

from pypsa.components.components import Components


class Loads(Components):
    """
    Loads components class.

    This class is used for load components. All functionality specific to
    loads is implemented here. Functionality for all components is implemented in
    the abstract base class.

    .. warning::
        This class is under ongoing development and will be subject to changes.
        It is not recommended to use this class outside of PyPSA.

    See Also
    --------
    pypsa.components.abstract.Components : Base class for all components.

    """

    base_attr = "p"
