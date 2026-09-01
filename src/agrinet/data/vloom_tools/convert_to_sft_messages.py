"""Pre-refactor compatibility entrypoint; use agrinet.data.vloom_conversion."""

from agrinet.data.vloom_conversion import *  # noqa: F401,F403
from agrinet.data.vloom_conversion import main


if __name__ == "__main__":
    main()
