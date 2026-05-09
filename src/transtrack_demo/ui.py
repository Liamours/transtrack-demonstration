try:
    from colorama import Fore, Style, init

    init(autoreset=True)
    WHITE = Fore.WHITE
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    DIM = Style.DIM
    BRIGHT = Style.BRIGHT
    RESET = Style.RESET_ALL
except Exception:
    WHITE = ""
    GREEN = ""
    YELLOW = ""
    DIM = ""
    BRIGHT = ""
    RESET = ""


def title():
    print()
    print(f"{GREEN}{BRIGHT}TransTrack Live Fatigue Detection{RESET}")
    print(f"{YELLOW}{'-' * 37}{RESET}")


def camera_menu(cameras):
    title()
    if cameras:
        print(f"{WHITE}Detected cameras:{RESET}")
        for camera in cameras:
            print(f"  {YELLOW}[{camera['index']}]{RESET} {WHITE}{camera['name']}{RESET}")
    else:
        print(f"{WHITE}No camera names detected. Try index 0 first.{RESET}")
    print()


def prompt_camera(default=0):
    raw = input(f"{GREEN}Camera index{RESET} {DIM}[{default}]{RESET}: ").strip()
    return int(raw) if raw else default


def run_info(camera_index, camera_name, backend, log_path):
    print()
    print(f"{GREEN}Running{RESET} {WHITE}press q in the camera window to stop{RESET}")
    print(f"{DIM}camera:{RESET} {YELLOW}{camera_index}{RESET} {WHITE}{camera_name}{RESET}")
    print(f"{DIM}backend:{RESET} {WHITE}{backend}{RESET}")
    print(f"{DIM}log:{RESET} {WHITE}{log_path}{RESET}")
    print()


def inference_line(result):
    scores = result.get("scores", {})
    score_text = " ".join(
        f"{WHITE}{name}{RESET}:{YELLOW}{score:.4f}{RESET}"
        for name, score in scores.items()
    )
    print(
        f"{GREEN}label={RESET}{WHITE}{result['label']}{RESET} "
        f"{GREEN}confidence={RESET}{YELLOW}{result['confidence']:.4f}{RESET} "
        f"{score_text}"
    )
