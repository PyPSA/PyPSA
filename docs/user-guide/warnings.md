<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Warnings Reference

PyPSA emits warnings to help catch common mistakes when building networks.
This page explains what they mean and how to resolve them.

!!! warning

  This page is a work in progress and not all warnings are documented yet.

## Misleading attribute name { #warning-attr-misleading }

> *"The attribute 'X' is a standard attribute for other components but not for Y.
> This could cause confusion and it should be renamed."*

This warning appears when you add a custom attribute to a component that happens
to be a standard attribute of a different component type. For example, adding
`s_nom` (a standard Line/Transformer attribute) to a Generator:

```py
n.add("Generator", "gen1", bus="bus1", s_nom=100)  # triggers warning
```

This usually indicates that the attribute was expected to exist for this component
but it does not. Additional attributes can be added to components, but they are
only descriptive and not used by PyPSA internally. Consider renaming it to avoid
confusion with the standard attribute.

!!! note
    This warning cannot be suppressed via options.

## Potential attribute typo { #warning-attr-typo }

> *"The attribute 'X' is not a standard attribute for Y. Did you mean 'Z'?"*

This warning appears when a custom attribute name is very similar to a standard
attribute (edit distance of 1), suggesting a typo. For example:

```python
n.add("Generator", "gen1", bus="bus1", p_no=100)  # did you mean 'p_nom'?
```

This warning can be suppressed if you intentionally use non-standard attribute
names that happen to be close to standard ones:

```python
pypsa.options.warnings.attribute_typos = False
```

See also [Warnings options](options.md#warnings-options).
