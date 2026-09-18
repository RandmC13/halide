import warnings

# colour-science warns at import time if matplotlib isn't installed, since it can't offer its
# plotting utilities — which this project never uses. The filter must be registered before any
# submodule below imports colour, since the warning fires during that very import.
warnings.filterwarnings(
    "ignore", message=r'"Matplotlib" related API features are not available', category=Warning
)
