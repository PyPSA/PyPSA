"""
Retrieve docstrings from runtime objects instead of static analysis (default).

See https://mkdocstrings.github.io/griffe/guide/users/how-to/selectively-inspect/
"""

import griffe

logger = griffe.get_logger("griffe_inspect_specific_objects")


class InspectSpecificObjects(griffe.Extension):
    """An extension to inspect just a few specific objects."""

    def __init__(self, objects: list[str]) -> None:
        self.objects = objects

    def on_instance(self, *, obj: griffe.Object, **kwargs) -> None:
        if obj.path not in self.objects:
            return
        logger.info("Using InspectSpecificObjects for %s", obj.path)

        try:
            runtime_obj = griffe.dynamic_import(obj.path)
        except ImportError as error:
            logger.warning("Could not import %s: %s", obj.path, error)
            return

        if obj.docstring:
            obj.docstring.value = runtime_obj.__doc__
        else:
            obj.docstring = griffe.Docstring(runtime_obj.__doc__)
