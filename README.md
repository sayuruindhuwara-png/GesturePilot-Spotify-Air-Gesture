# Spotify Air Gesture for PC

A Windows-first webcam controller inspired by Spotify Air Gesture. It uses MediaPipe hand poses and landmarks and requires a short hold for playback actions.

## Gestures

- **Open palm:** hold your palm open and trace one broad circle with your palm. Clockwise from your viewpoint lowers Windows master volume by 15 percentage points; counterclockwise raises it by 15. Keep your palm open while tracing.
- **Pinch and drag left:** next track.
- **Pinch and drag right:** previous track.
- **Victory (index and middle fingers):** toggle Spotify between maximized and restored window sizes.
- **Open palm without a circle:** neutral and re-arm gesture controls.
- **Closed fist held briefly:** play/pause. Right-hand fist recognition includes a landmark-based fallback.
- **Thumbs up/down held briefly:** Spotify Like/Dislike Song shortcut.
- **No hand:** pauses gesture control; it does not arm the controller.

Pinch drags trigger one action per pinch: move the hand in the desired direction, then release and pinch again. Spotify's combined **Like/Dislike Song** shortcut (`Alt+Shift+B`) is used for both thumbs gestures, so Spotify decides whether that command likes or unlikes the current song.

## Customize controls

Edit `gesture_config.json` to remap built-in poses, then restart the app. `media_gestures` maps a pose name to an action. `pinch_directions` maps left/right/up/down movement to actions. Available actions are `play`, `pause`, `play_pause`, `next_track`, `previous_track`, `like_toggle`, `volume_up`, `volume_down`, `mute`, `scroll_up`, `scroll_down`, `maximize_restore`, and `none`. You can also use `key:space` or `key:ctrl+alt+m` to send a custom keyboard shortcut to Spotify. `Unknown` and `Transition` are reserved neutral gestures.

Keep Spotify open. The app downloads the MediaPipe gesture model once on first launch and stores it in `assets`; internet is needed for that initial download. It requests 1280x720 and minimum zoom when supported by the camera.

## Setup (Windows)

Use Python 3.11. In PowerShell, from this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. Press **Q** or **Esc** while the preview is focused to quit. Move the mouse to the top-left corner for PyAutoGUI's emergency stop.
















