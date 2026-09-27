# Spotify Air Gesture for PC

A Windows-first webcam controller inspired by Spotify Air Gesture. It uses MediaPipe hand poses and landmarks, with a short hold required for playback actions.

## Gestures

- **Open palm:** Keep your palm facing the camera and trace one broad circle with your index fingertip. Keep the circle centered in the camera view, with your hand near the middle of the frame. Clockwise lowers Windows master volume by 15 percentage points; counterclockwise raises it by 15. Keep your palm open while tracing.
- **Pinch and drag left:** Next track.
- **Pinch and drag right:** Previous track.
- **Victory (index and middle fingers):** Toggle Spotify between maximized and restored window sizes.
- **Open palm without a circle:** Neutral gesture that re-arms the controls.
- **Closed fist held briefly:** Play or pause. Right-hand fist recognition includes a landmark-based fallback.
- **Thumbs up or thumbs down held briefly:** Send Spotify’s Like/Dislike Song shortcut.
- **No hand:** Pauses gesture control; it does not arm the controller.

Pinch drags trigger one action per pinch: move your hand in the desired direction, then release and pinch again. Both thumbs gestures use Spotify’s combined **Like/Dislike Song** shortcut (`Alt+Shift+B`), so Spotify decides whether to like or unlike the current song.

## Customize controls

Edit `gesture_config.json` to remap built-in poses, then restart the app. `media_gestures` maps a pose name to an action. `pinch_directions` maps left, right, up, and down movement to actions.

Available actions are `play`, `pause`, `play_pause`, `next_track`, `previous_track`, `like_toggle`, `volume_up`, `volume_down`, `mute`, `scroll_up`, `scroll_down`, `maximize_restore`, and `none`. You can also use `key:space` or `key:ctrl+alt+m` to send a custom keyboard shortcut to Spotify. `Unknown` and `Transition` are reserved neutral gestures.

Keep Spotify open while using the controller. On first launch, the app downloads the MediaPipe gesture model and stores it in `assets`; an internet connection is needed for this initial download. The app requests 1280×720 resolution and minimum camera zoom when supported.

## Setup (Windows)

Use Python 3.11. In PowerShell, run these commands from the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, run Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass in that terminal, then activate the environment again.
Press Q or Esc while the preview window is focused to quit. Move the mouse to the top-left corner to trigger PyAutoGUI’s emergency stop.
