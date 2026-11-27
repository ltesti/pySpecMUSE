"""Sphinx configuration for the installed pySpecMUSE package."""

from importlib.metadata import version as distribution_version

project = "pySpecMUSE"
author = "The pySpecMUSE authors"
release = distribution_version("pySpecMUSE")

extensions = ["sphinx.ext.autodoc"]
exclude_patterns = ["_build"]
html_theme = "alabaster"

# Astropy is an external API. Keep builds independent of its online inventory.
nitpick_ignore = [("py:class", "astropy.table.table.Table")]
