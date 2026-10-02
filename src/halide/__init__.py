import warnings

# colour-science warns at import time if matplotlib isn't installed, since it can't offer its
# plotting utilities — which this project never uses. The filter must be registered before any
# submodule below imports colour, since the warning fires during that very import.
COLOUR_MATPLOTLIB_MESSAGE = r'"Matplotlib" related API features are not available'


def ignore_colour_matplotlib_warning() -> None:
    """Ignore that warning. Call it again inside any `warnings.simplefilter("always")` block,
    which would otherwise override this package-level filter."""
    warnings.filterwarnings("ignore", message=COLOUR_MATPLOTLIB_MESSAGE, category=Warning)


ignore_colour_matplotlib_warning()
