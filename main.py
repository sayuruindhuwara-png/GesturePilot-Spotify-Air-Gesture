import ctypes
import json
import math
import time
from collections import Counter, deque
from ctypes import wintypes
from pathlib import Path
from urllib.request import urlopen

import cv2
import mediapipe as mp
import pyautogui
import pygetwindow as gw

# Webcam Spotify controller: pretrained static gesture recognition + tracked motion.
pyautogui.FAILSAFE = True  # Move the mouse to the top-left corner to stop control.
pyautogui.PAUSE = 0

CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
MARGIN = 0.08
SMOOTHING = 0.28
SCROLL_GAIN = 3.5
NEUTRAL_REARM_SECONDS = 0.0
PINCH_RELEASE_SECONDS = 0.25
GESTURE_SCORE_MIN = 0.65
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
MODEL_PATH = Path(__file__).resolve().parent / "assets" / "gesture_recognizer.task"
PREVIEW_TITLE = "GesturePilot — Spotify Air Gesture PC"
CONFIG_PATH = Path(__file__).resolve().parent / "gesture_config.json"
DEFAULT_CONFIG = {
    "hold_seconds": 0.55,
    "swipe_distance": 0.10,
    "media_gestures": {
        "Closed_Fist": "play_pause",
        "Thumb_Up": "like_toggle",
        "Thumb_Down": "like_toggle",
        "Victory": "maximize_restore",
    },
    "pinch_directions": {
        "right": "previous_track",
        "left": "next_track",
        "up": "scroll_up",
        "down": "scroll_down",
    },
}
ACTION_LABELS = {
    "play_pause": "PLAY / PAUSE",
    "play": "PLAY",
    "pause": "PAUSE",
    "next_track": "NEXT TRACK",
    "previous_track": "PREVIOUS TRACK",
    "like_toggle": "LIKE / DISLIKE SONG",
    "volume_up": "VOLUME UP",
    "volume_down": "VOLUME DOWN",
    "mute": "MUTE",
    "scroll_up": "SCROLL UP",
    "scroll_down": "SCROLL DOWN",
    "none": "DISABLED",
    "maximize_restore": "MAXIMIZE / RESTORE SPOTIFY",
}

screen_w, screen_h = pyautogui.size()


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def angle_degrees(a, b, c):
    """Angle ABC using normalized hand-landmark coordinates."""
    ab = (a.x - b.x, a.y - b.y, a.z - b.z)
    cb = (c.x - b.x, c.y - b.y, c.z - b.z)
    dot = sum(x * y for x, y in zip(ab, cb))
    norm_ab = math.sqrt(sum(x * x for x in ab))
    norm_cb = math.sqrt(sum(x * x for x in cb))
    cosine = max(-1.0, min(1.0, dot / max(norm_ab * norm_cb, 1e-6)))
    return math.degrees(math.acos(cosine))


def looks_like_closed_fist(landmarks):
    """Use finger curl and thumb position when MediaPipe misses a right fist."""
    curled = 0
    for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12),
                               (13, 14, 15, 16), (17, 18, 19, 20)):
        if (angle_degrees(landmarks[mcp], landmarks[pip], landmarks[dip]) < 160
                and angle_degrees(landmarks[pip], landmarks[dip], landmarks[tip]) < 170):
            curled += 1
    center_x = sum(landmarks[i].x for i in (0, 5, 9, 13, 17)) / 5
    center_y = sum(landmarks[i].y for i in (0, 5, 9, 13, 17)) / 5
    palm_width = max(distance(landmarks[5], landmarks[17]), 1e-4)
    thumb_tucked = math.hypot(landmarks[4].x - center_x, landmarks[4].y - center_y) < palm_width * 1.35
    return curled >= 3 and thumb_tucked


def analyze_circle(points):
    """Return (in_progress, direction) for a tracked circular index-finger path."""
    if len(points) < 10:
        return False, 0
    a, b, c = points[0], points[len(points) // 3], points[(2 * len(points)) // 3]
    determinant = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(determinant) < 1e-5:
        return False, 0
    aa, bb, cc = a[0] ** 2 + a[1] ** 2, b[0] ** 2 + b[1] ** 2, c[0] ** 2 + c[1] ** 2
    center_x = (aa * (b[1] - c[1]) + bb * (c[1] - a[1]) + cc * (a[1] - b[1])) / determinant
    center_y = (aa * (c[0] - b[0]) + bb * (a[0] - c[0]) + cc * (b[0] - a[0])) / determinant
    vectors = [(x - center_x, y - center_y) for x, y in points]
    radii = [math.hypot(x, y) for x, y in vectors]
    radius = sum(radii) / len(radii)
    if radius < 0.035 or radius > 0.25:
        return False, 0
    radial_error = sum(abs(value - radius) for value in radii) / len(radii)
    if radial_error > radius * 0.30:
        return False, 0
    rotation = 0.0
    for (ax, ay), (bx, by) in zip(vectors, vectors[1:]):
        step = math.atan2(ax * by - ay * bx, ax * bx + ay * by)
        if abs(step) > math.pi / 2:
            return False, 0
        rotation += step
    in_progress = abs(rotation) >= 0.8
    closure = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
    # A full 360-degree loop is difficult to trace accurately in front of a
    # webcam. Accept a clean ~250-degree loop with a generous closing margin.
    if in_progress and abs(rotation) >= math.pi * 1.4 and closure <= radius * 1.1:
        return True, 1 if rotation > 0 else -1
    return in_progress, 0


def valid_action(action):
    if not isinstance(action, str):
        return False
    return action in ACTION_LABELS or (
        action.startswith("key:") and all(part.strip() for part in action[4:].split("+"))
    )


def load_gesture_config():
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8")
    try:
        user_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read {CONFIG_PATH.name}: {exc}") from exc
    if not isinstance(user_config, dict):
        raise RuntimeError(f"{CONFIG_PATH.name} must contain a JSON object.")
    media_overrides = user_config.get("media_gestures", {})
    pinch_overrides = user_config.get("pinch_directions", {})
    if not isinstance(media_overrides, dict):
        raise RuntimeError("media_gestures must be a JSON object of gesture names to actions.")
    if not isinstance(pinch_overrides, dict):
        raise RuntimeError("pinch_directions must be a JSON object.")

    config = {
        "hold_seconds": user_config.get("hold_seconds", DEFAULT_CONFIG["hold_seconds"]),
        "swipe_distance": user_config.get("swipe_distance", DEFAULT_CONFIG["swipe_distance"]),
        "media_gestures": {**DEFAULT_CONFIG["media_gestures"], **media_overrides},
        "pinch_directions": {**DEFAULT_CONFIG["pinch_directions"], **pinch_overrides},
    }
    try:
        config["hold_seconds"] = float(config["hold_seconds"])
        config["swipe_distance"] = float(config["swipe_distance"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("hold_seconds and swipe_distance must be numbers.") from exc
    if not 0.15 <= config["hold_seconds"] <= 3.0:
        raise RuntimeError("hold_seconds must be between 0.15 and 3.0.")
    if not 0.03 <= config["swipe_distance"] <= 0.5:
        raise RuntimeError("swipe_distance must be between 0.03 and 0.5.")

    actions = config["media_gestures"]
    if not isinstance(actions, dict):
        raise RuntimeError("media_gestures must be a JSON object of gesture names to actions.")
    for gesture, action in actions.items():
        gesture_name = gesture.split(":", 1)[-1] if isinstance(gesture, str) else ""
        if not isinstance(gesture, str) or gesture_name in ("Unknown", "Transition"):
            raise RuntimeError("Unknown and Transition are reserved neutral gestures.")
        if not valid_action(action):
            raise RuntimeError(f"Unsupported action for {gesture}: {action}")

    directions = config["pinch_directions"]
    if not isinstance(directions, dict):
        raise RuntimeError("pinch_directions must be a JSON object.")
    for direction in ("left", "right", "up", "down"):
        if direction not in directions or not valid_action(directions[direction]):
            raise RuntimeError(f"pinch_directions.{direction} must be a supported action.")
    return config


def run_control_action(action):
    if action == "play":
        return send_spotify_media_key(0xFA)
    if action == "play_pause":
        return toggle_spotify_playback()
    if action == "pause":
        return send_spotify_media_key(0xB3)
    if action == "next_track":
        return send_spotify_media_key(0xB0)
    if action == "previous_track":
        return send_spotify_media_key(0xB1)
    if action == "like_toggle":
        return send_spotify_shortcut("alt", "shift", "b")
    if action == "maximize_restore":
        hwnd = find_spotify_window()
        if not hwnd:
            return False
        user32 = ctypes.windll.user32
        user32.ShowWindow(hwnd, 9 if user32.IsZoomed(hwnd) else 3)
        user32.SetForegroundWindow(hwnd)
        return True
    if action == "volume_up":
        return adjust_system_volume(0.15)
    elif action == "volume_down":
        return adjust_system_volume(-0.15)
    elif action == "mute":
        key = 0xAD
    elif action in ("scroll_up", "scroll_down"):
        pyautogui.scroll(5 if action == "scroll_up" else -5)
        return True
    elif action.startswith("key:"):
        return send_spotify_shortcut(*[part.strip() for part in action[4:].split("+")])
    elif action == "none":
        return True
    else:
        return False
    user32 = ctypes.windll.user32
    user32.keybd_event(key, 0, 0, 0)
    user32.keybd_event(key, 0, 0x0002, 0)
    return True


def adjust_system_volume(delta):
    """Change Windows master output volume by an exact percentage-point step."""
    try:
        from pycaw.pycaw import AudioUtilities

        endpoint = AudioUtilities.GetSpeakers().EndpointVolume
        current = endpoint.GetMasterVolumeLevelScalar()
        endpoint.SetMasterVolumeLevelScalar(max(0.0, min(1.0, current + delta)), None)
        return True
    except Exception as exc:
        print(f"Could not adjust system volume: {exc}")
        return False


def ensure_model():
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 1_000_000:
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = MODEL_PATH.with_suffix(".download")
    print("Downloading MediaPipe's hand gesture model (one time)...")
    try:
        with urlopen(MODEL_URL, timeout=60) as response, partial.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        if partial.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded model file is incomplete.")
        partial.replace(MODEL_PATH)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the gesture model. Connect to the internet once, "
            "then run the app again."
        ) from exc


def find_spotify_window():
    """Find Spotify by process name because its title changes with the song."""
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
    )
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    found = []
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        handle = kernel32.OpenProcess(0x1000, False, pid.value)
        if not handle:
            return True
        try:
            path = ctypes.create_unicode_buffer(32768)
            size = wintypes.DWORD(len(path))
            if kernel32.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(size)):
                if Path(path.value).name.casefold() == "spotify.exe":
                    rect = wintypes.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    title = ctypes.create_unicode_buffer(512)
                    user32.GetWindowTextW(hwnd, title, len(title))
                    area = max(0, rect.right - rect.left) * max(0, rect.bottom - rect.top)
                    if title.value.strip() and area > 100_000:
                        found.append((area, hwnd))
        finally:
            kernel32.CloseHandle(handle)
        return True

    user32.EnumWindows(enum_proc(visit), 0)
    return max(found, default=(0, None), key=lambda item: item[0])[1]


def send_spotify_shortcut(*keys):
    hwnd = find_spotify_window()
    if not hwnd:
        return False
    user32 = ctypes.windll.user32
    try:
        user32.ShowWindow(hwnd, 9)
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.12)
        if user32.GetForegroundWindow() != hwnd:
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            # A click in the title-bar area reliably activates Spotify without
            # clicking playback controls or changing the current track.
            pyautogui.click(rect.left + 100, rect.top + 12)
            time.sleep(0.15)
        if user32.GetForegroundWindow() != hwnd:
            return False
        if len(keys) == 1:
            pyautogui.press(keys[0])
        else:
            pyautogui.hotkey(*keys)
        return True
    except Exception:
        return False
    finally:
        time.sleep(0.08)
        previews = gw.getWindowsWithTitle(PREVIEW_TITLE)
        if previews:
            try:
                previews[0].activate()
            except Exception:
                pass


def toggle_spotify_playback():
    """Send the Windows media play/pause key, which works without app focus."""
    if not find_spotify_window():
        return False
    user32 = ctypes.windll.user32
    vk_media_play_pause = 0xB3
    keyup = 0x0002
    user32.keybd_event(vk_media_play_pause, 0, 0, 0)
    user32.keybd_event(vk_media_play_pause, 0, keyup, 0)
    return True


def send_spotify_media_key(virtual_key):
    """Use Windows' media transport key for play, previous, or next."""
    if not find_spotify_window():
        return False
    user32 = ctypes.windll.user32
    user32.keybd_event(virtual_key, 0, 0, 0)
    user32.keybd_event(virtual_key, 0, 0x0002, 0)
    return True


def top_gesture(result):
    if not result.gestures or not result.gestures[0]:
        return "Unknown", 0.0
    category = result.gestures[0][0]
    name = category.category_name or "Unknown"
    if name == "None":
        name = "Unknown"
    return name, float(category.score or 0.0)


def main():
    config = load_gesture_config()
    hold_seconds = config["hold_seconds"]
    swipe_distance = config["swipe_distance"]
    media_gestures = config["media_gestures"]
    pinch_directions = config["pinch_directions"]
    action_gestures = tuple(media_gestures)
    ensure_model()
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check CAMERA_INDEX or camera permissions.")
    zoom_supported = cap.set(cv2.CAP_PROP_ZOOM, 0)
    zoom_status = "minimum zoom requested" if zoom_supported else "zoom control unavailable"
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    vision = mp.tasks.vision
    options = vision.GestureRecognizerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.65,
    )

    smoother = None
    previous_scroll_y = None
    gesture_history = deque(maxlen=5)
    circle_points = deque(maxlen=60)
    candidate = None
    candidate_since = 0.0
    action_latched = False
    pinch_active = False
    pinch_origin_x = None
    pinch_origin_y = None
    pinch_swipe_latched = False
    pinch_release_since = None
    neutral_since = None
    status = "Show one hand to begin"
    start_time = time.monotonic()

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                status = "Camera frame unavailable"
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)
            detected, score = top_gesture(result)
            landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
            hand_side = "Unknown"
            if result.handedness and result.handedness[0]:
                hand_side = result.handedness[0][0].category_name or "Unknown"
            if landmarks:
                # Draw tracked landmarks with OpenCV, independent of the recognizer API.
                for a, b in mp.solutions.hands.HAND_CONNECTIONS:
                    p1, p2 = landmarks[a], landmarks[b]
                    pt1 = (int(p1.x * frame.shape[1]), int(p1.y * frame.shape[0]))
                    pt2 = (int(p2.x * frame.shape[1]), int(p2.y * frame.shape[0]))
                    cv2.line(frame, pt1, pt2, (80, 220, 130), 2)
                for p in landmarks:
                    cv2.circle(frame, (int(p.x * frame.shape[1]), int(p.y * frame.shape[0])), 3, (20, 210, 255), -1)

                if score < GESTURE_SCORE_MIN:
                    detected = "Unknown"
                if hand_side == "Right" and looks_like_closed_fist(landmarks):
                    detected, score = "Closed_Fist", max(score, 0.90)
                gesture_history.append(detected)
                counts = Counter(gesture_history)
                best_count = max(counts.values())
                stable = next(name for name in reversed(gesture_history) if counts[name] == best_count)
                if best_count < 3:
                    stable = "Transition"
                action_key = stable
                now = time.monotonic()
                pinch_ratio = distance(landmarks[4], landmarks[8]) / max(distance(landmarks[5], landmarks[17]), 0.001)
                if pinch_active:
                    if pinch_ratio >= 0.76:
                        if pinch_release_since is None:
                            pinch_release_since = now
                        released = now - pinch_release_since >= PINCH_RELEASE_SECONDS
                        is_pinch = not released
                    else:
                        pinch_release_since = None
                        is_pinch = True
                else:
                    is_pinch = pinch_ratio < 0.50 and stable not in (
                        "Open_Palm", "Closed_Fist", "Thumb_Up", "Thumb_Down"
                    )
                if action_key in action_gestures:
                    neutral_since = None
                    if action_key != candidate:
                        candidate, candidate_since = action_key, now
                elif stable in ("Open_Palm", "Unknown"):
                    candidate = None
                    if neutral_since is None:
                        neutral_since = now
                    elif now - neutral_since >= NEUTRAL_REARM_SECONDS:
                        action_latched = False
                elif stable != "Transition":
                    candidate = None
                    neutral_since = None
                else:
                    neutral_since = None

                x, y = landmarks[8].x, landmarks[8].y
                if not is_pinch and stable != "Transition":
                    pinch_active = False
                    pinch_origin_x = None
                    pinch_origin_y = None
                    pinch_swipe_latched = False
                    pinch_release_since = None
                if is_pinch:
                    # Track the palm center so the swipe direction is stable for either hand.
                    pinch_x = landmarks[9].x
                    pinch_y = landmarks[9].y
                    if not pinch_active:
                        pinch_active = True
                        pinch_origin_x = pinch_x
                        pinch_origin_y = pinch_y
                        pinch_release_since = None
                    dx = pinch_x - pinch_origin_x
                    dy = pinch_y - pinch_origin_y
                    if not pinch_swipe_latched and max(abs(dx), abs(dy)) >= swipe_distance:
                        direction = "right" if abs(dx) >= abs(dy) and dx > 0 else (
                            "left" if abs(dx) >= abs(dy) else ("up" if dy < 0 else "down")
                        )
                        action = pinch_directions[direction]
                        if run_control_action(action):
                            status = ACTION_LABELS.get(action, action.removeprefix("key:").upper())
                        else:
                            status = "Spotify app not found or control failed"
                        pinch_swipe_latched = True
                    elif pinch_swipe_latched:
                        status = "Release pinch before the next gesture"
                    else:
                        status = "Pinch and drag: left/right track, up/down scroll"
                elif action_key in action_gestures:
                    held = now - candidate_since
                    action = media_gestures[action_key]
                    if action_latched:
                        status = "Release gesture to re-arm"
                    elif held >= hold_seconds:
                        if run_control_action(action):
                            status = ACTION_LABELS.get(action, action.removeprefix("key:").upper())
                        else:
                            status = "Spotify app not found or control failed"
                        action_latched = True
                    else:
                        label = ACTION_LABELS.get(action, action.removeprefix("key:").upper())
                        status = f"Hold for {label.lower()}: {max(0.0, hold_seconds-held):.1f}s"
                elif stable in ("Pointing_Up", "Open_Palm"):
                    circle_points.append((x, y, now))
                    while circle_points and now - circle_points[0][2] > 1.8:
                        circle_points.popleft()
                    circle_in_progress, rotation = analyze_circle(
                        [(point[0], point[1]) for point in circle_points]
                    )
                    if rotation:
                        # The preview is mirrored: clockwise from the user's
                        # viewpoint is negative angular movement in frame coords.
                        action = "volume_down" if rotation < 0 else "volume_up"
                        if run_control_action(action):
                            status = ACTION_LABELS[action]
                        else:
                            status = "Spotify app not found or volume control failed"
                        circle_points.clear()
                        previous_scroll_y = None
                        circle_in_progress = True
                    if circle_in_progress:
                        status = "CIRCLE — adjusting volume"
                    else:
                        pose_label = "OPEN PALM" if stable == "Open_Palm" else "POINT UP"
                        status = f"{pose_label} — circle clockwise for volume up, counterclockwise for down"
                else:
                    circle_points.clear()
                    if stable == "Victory":
                        center_y = landmarks[9].y
                        if previous_scroll_y is not None:
                            delta = previous_scroll_y - center_y
                            if abs(delta) > 0.008:
                                pyautogui.scroll(int(max(-12, min(12, delta * SCROLL_GAIN * 100))))
                        previous_scroll_y = center_y
                        status = "VICTORY — move hand to scroll"
                    else:
                        previous_scroll_y = None

                    if stable in ("Open_Palm", "Unknown"):
                        if neutral_since is not None and now - neutral_since >= NEUTRAL_REARM_SECONDS:
                            status = "NEUTRAL — ready"
                        else:
                            status = "Show open palm to reset gestures"
                    elif stable == "Transition":
                        status = "Hold gesture steady — checking"
                    elif stable not in ("Pointing_Up", "Victory"):
                        status = "Pinch, fist, or thumb gesture"
            else:
                gesture_history.clear()
                previous_scroll_y = None
                candidate = None
                pinch_active = False
                pinch_origin_x = None
                pinch_origin_y = None
                if neutral_since is None:
                    neutral_since = time.monotonic()
                elif time.monotonic() - neutral_since >= NEUTRAL_REARM_SECONDS:
                    action_latched = False
                status = "No hand — control paused"

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 94), (25, 25, 25), -1)
            cv2.putText(frame, status, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (100, 240, 150), 2)
            cv2.putText(frame, f"Detected: {detected}   Hand: {hand_side}   Confidence: {score:.0%}   Q/Esc: quit   Top-left: emergency stop",
                        (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1)
            cv2.putText(frame, f"Camera: {actual_width}x{actual_height}   {zoom_status}",
                        (18, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 210, 230), 1)
            cv2.imshow(PREVIEW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
