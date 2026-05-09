import site
import sys


def _remove_user_site_packages():
    paths = {site.getusersitepackages()}
    paths.update(site.getsitepackages() if hasattr(site, "getsitepackages") else [])
    user_roaming = "\\AppData\\Roaming\\Python\\"

    sys.path[:] = [
        path for path in sys.path
        if user_roaming.lower() not in path.lower()
    ]


_remove_user_site_packages()

from src.transtrack_demo.live_inference import main


if __name__ == "__main__":
    main()
