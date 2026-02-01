# Hand Gesture Controller 🖐️🖱️

A futuristic, "Minority Report" inspired interface that transforms your hand into a virtual mouse. Powered by **Python**, **MediaPipe**, and **OpenCV**, this tool delivers a seamless user experience.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-green?style=for-the-badge&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-red?style=for-the-badge&logo=opencv&logoColor=white)

## ✨ Capabilities

Experience a new way to interact with your digital world:

-   **👆 Precision Tracking**: High-fidelity index finger tracking with EMA smoothing for a jitter-free experience.
-   **🖱️ Intuitive Gestures**:
    -   **Move**: Simply point your **Index Finger**.
    -   **Left Click**: Pinch **Index Finger** + **Thumb**.
    -   **Right Click**: Pinch **Middle Finger** + **Thumb**.
    -   **Scroll Mode**: Raise both **Index** & **Middle Fingers** and move your hand vertically.
-   **🛡️ FailSafe Protocol**: Instantly terminate the session by moving your hand to any screen corner.
-   **📊 Live HUD**: Real-time FPS monitoring and current gesture mode display.

## 🛠️ Quick Start

### 1. Installation

Clone the repo and set up your environment:

```bash
# Clone
git clone https://github.com/Arnav-SAI/Hand-Gesture-Controller
cd hand-gesture-controls

# Virtual Env (Recommended)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install Deps
pip install -r requirements.txt
```

### 2. Launch

Ignite the engine:

```bash
python main.py
```

### 3. Permissions (macOS)

For the magic to happen, you need to grant a few permissions:
1.  **Camera**: To see your hand.
2.  **Accessibility**: To control the mouse. *System Settings -> Privacy & Security -> Accessibility*.
    *   *Tip: If the cursor moves but clicks fail, checking Accessibility permissions usually fixes it.*

## ⚙️ Customization

Tailor the experience by modifying the constants in `main.py`:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SMOOTHING_FACTOR` | `5` | Higher values = smoother cursor but slightly more latency. |
| `FRAME_MARGIN` | `100` | The invisible "deadzone" box around your webcam feed. |
| `CLICK_THRESHOLD` | `40` | Sensitivity for pinch gestures. Increase if clicks are hard to trigger. |
| `DOUBLE_CLICK_COOLDOWN` | `0.5` | Minimum time (in seconds) between clicks to prevent accidental double-clicks. |

## 🎮 Controls Summary

| Gesture | Action |
| :--- | :--- |
| **Index Up** | Move Cursor |
| **Index + Thumb Pinch** | Left Click |
| **Middle + Thumb Pinch** | Right Click |
| **Index + Middle Up** | Scroll Mode |
| **All Fingers Down/Fist** | Idle (No Action) |

---


