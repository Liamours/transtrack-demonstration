import site
import sys


def _remove_user_site_packages():
    user_site = site.getusersitepackages().lower()
    sys.path[:] = [
        path for path in sys.path
        if path.lower() != user_site
    ]


_remove_user_site_packages()

from src.transtrack_demo.live_inference import main


if __name__ == "__main__":
    main()
