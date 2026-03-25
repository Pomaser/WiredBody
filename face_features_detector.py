import os
import sys

# Add conda env DLL directories to PATH so libs load even without activation
_env_root = os.path.dirname(sys.executable)
for _dll_dir in [
    os.path.join(_env_root, "Library", "bin"),
    os.path.join(_env_root, "Library", "mingw-w64", "bin"),
    os.path.join(_env_root, "Library", "usr", "bin"),
]:
    if os.path.isdir(_dll_dir) and _dll_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _dll_dir + os.pathsep + os.environ["PATH"]

import ctypes
import math
import time
import winsound
import cv2
import argparse
import mediapipe as mp

GLASSES_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glasses.png")
FUCK_OFF_SOUND   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fuck_off.wav")
FUCK_OFF_COOLDOWN = 3.0   # sekundy mezi opakovaným přehráním

_last_fuck_off = 0.0
GLASSES_SCALE    = 0.75   # 1.0 = šířka obličeje, 0.85 = 85 % šířky obličeje
GLASSES_OFFSET_Y = 25     # kladné = posun dolů (px v původním rozlišení kamery)
GLASSES_SMOOTH   = 0.25   # EMA alpha: 0 = max plynulost, 1 = bez vyhlazení

# Stav pro EMA vyhlazení brýlí
_g_smooth = {"gx": None, "gy": None, "angle": 0.0}

def _ema(prev, cur, alpha):
    """Exponenciální klouzavý průměr."""
    return cur if prev is None else alpha * cur + (1.0 - alpha) * prev

def load_glasses():
    img = cv2.imread(GLASSES_PATH, cv2.IMREAD_UNCHANGED)  # načti i alfa kanál
    if img is None:
        print(f"[WARNING] Cannot load glasses: {GLASSES_PATH}")
        return None, None
    if img.shape[2] == 4:
        mask = img[:, :, 3]          # alfa kanál = maska
        img  = img[:, :, :3]         # BGR bez alfy
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    return img, mask


def overlay_image(bg, overlay, mask, x, y):
    oh, ow = overlay.shape[:2]
    bh, bw = bg.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + ow), min(bh, y + oh)
    if x2 <= x1 or y2 <= y1:
        return
    ov = overlay[y1 - y:y2 - y, x1 - x:x2 - x]
    mk = mask[y1 - y:y2 - y, x1 - x:x2 - x]
    bg[y1:y2, x1:x2][mk > 0] = ov[mk > 0]


def get_screen_size():
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

# ---------------------------------------------------------------------------
# Haar cascade classifiers (face, eyes, mouth)
# ---------------------------------------------------------------------------
FACE_CASCADE  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
MOUTH_CASCADE = cv2.data.haarcascades + "haarcascade_smile.xml"

# ---------------------------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------------------------
mp_hands      = mp.solutions.hands
mp_face_mesh  = mp.solutions.face_mesh
mp_draw       = mp.solutions.drawing_utils
mp_styles     = mp.solutions.drawing_styles


# Colours for the wired hand
LANDMARK_COLOR   = mp_draw.DrawingSpec(color=(0, 255, 180), thickness=2, circle_radius=4)
CONNECTION_COLOR = mp_draw.DrawingSpec(color=(0, 180, 255), thickness=2)


def load_cascades():
    face_cc  = cv2.CascadeClassifier(FACE_CASCADE)
    mouth_cc = cv2.CascadeClassifier(MOUTH_CASCADE)

    for name, cc in [("face", face_cc), ("mouth", mouth_cc)]:
        if cc.empty():
            print(f"[ERROR] Could not load {name} cascade.")
            sys.exit(1)

    return face_cc, mouth_cc


def eyes_from_mesh(mesh_results, frame_w, frame_h):
    """Vrátí ((rx,ry), (lx,ly), eye_dist) nebo None. Funguje nezávisle na Haar."""
    if not mesh_results or not mesh_results.multi_face_landmarks:
        return None
    lm = mesh_results.multi_face_landmarks[0].landmark
    def pt(idx):
        return int(lm[idx].x * frame_w), int(lm[idx].y * frame_h)
    # pravé oko: vnější 33, vnitřní 133  |  levé oko: vnitřní 362, vnější 263
    rx = (pt(33)[0] + pt(133)[0]) // 2;  ry = (pt(33)[1] + pt(133)[1]) // 2
    lx = (pt(362)[0] + pt(263)[0]) // 2; ly = (pt(362)[1] + pt(263)[1]) // 2
    eye_dist = int(math.hypot(lx - rx, ly - ry))
    return (rx, ry), (lx, ly), eye_dist


def _draw_glasses(frame, mesh_results, glasses_img, glasses_mask):
    """Překryje brýle nad oči čistě z Face Mesh — nezávislé na Haar."""
    if glasses_img is None or glasses_mask is None:
        return
    h_fr, w_fr = frame.shape[:2]
    eyes = eyes_from_mesh(mesh_results, w_fr, h_fr)

    # Aktualizuj EMA jen když Face Mesh vidí oči
    if eyes is not None:
        (rx, ry), (lx, ly), eye_dist = eyes
        # Šířka brýlí z rozestupu očí (outer–outer vzdálenost * scale)
        g_w = int(eye_dist * 3.5 * GLASSES_SCALE)
        g_w = max(g_w, 40)
        angle = -math.degrees(math.atan2(ly - ry, lx - rx))
        cx = (rx + lx) // 2
        cy = (ry + ly) // 2
        g_h = int(g_w * glasses_img.shape[0] / glasses_img.shape[1])
        raw_gx = cx - g_w // 2
        raw_gy = cy - int(g_h * 0.35) + GLASSES_OFFSET_Y   # 35 % výšky = pozice očí v brýlích
        _g_smooth["gx"]    = _ema(_g_smooth["gx"],    raw_gx, GLASSES_SMOOTH)
        _g_smooth["gy"]    = _ema(_g_smooth["gy"],    raw_gy, GLASSES_SMOOTH)
        _g_smooth["angle"] = _ema(_g_smooth["angle"], angle,   GLASSES_SMOOTH)

    # Kresli pokud máme platnou pozici (i fallback z minulých snímků)
    if _g_smooth["gx"] is None:
        return

    # Odhadni aktuální g_w z posledního eye_dist nebo z EMA gx
    if eyes is not None:
        _, _, eye_dist = eyes
        g_w = int(eye_dist * 3.5 * GLASSES_SCALE)
        g_w = max(g_w, 40)
    else:
        g_w = int(abs(_g_smooth.get("gw", 80)))
    _g_smooth["gw"] = g_w

    angle = _g_smooth["angle"]
    g_h = int(g_w * glasses_img.shape[0] / glasses_img.shape[1])
    g_img  = cv2.resize(glasses_img,  (g_w, g_h))
    g_mask = cv2.resize(glasses_mask, (g_w, g_h))

    rad = math.radians(abs(angle))
    pad_x = int(g_h * math.sin(rad) / 2) + 1
    pad_y = int(g_w * math.sin(rad) / 2) + 1
    g_img  = cv2.copyMakeBorder(g_img,  pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
    g_mask = cv2.copyMakeBorder(g_mask, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
    M = cv2.getRotationMatrix2D((g_w // 2 + pad_x, g_h // 2 + pad_y), angle, 1.0)
    g_img  = cv2.warpAffine(g_img,  M, (g_w + 2*pad_x, g_h + 2*pad_y))
    g_mask = cv2.warpAffine(g_mask, M, (g_w + 2*pad_x, g_h + 2*pad_y))

    overlay_image(frame, g_img, g_mask,
                  int(_g_smooth["gx"]) - pad_x,
                  int(_g_smooth["gy"]) - pad_y)


def detect_face_features(frame, face_cc, mouth_cc, mesh_results=None, glasses_img=None, glasses_mask=None, show_text=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Brýle řídí Face Mesh — nezávislé na Haar
    _draw_glasses(frame, mesh_results, glasses_img, glasses_mask)

    faces = face_cc.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    for (fx, fy, fw, fh) in faces:
        face_gray  = gray[fy:fy + fh, fx:fx + fw]
        face_color = frame[fy:fy + fh, fx:fx + fw]

        # Mouth — lower 50 % of face
        mouth_y0 = int(fh * 0.50)
        mouths = mouth_cc.detectMultiScale(
            face_gray[mouth_y0:, :], scaleFactor=1.5, minNeighbors=20, minSize=(30, 15)
        )
        if show_text:
            for (mx, my, mw, mh) in mouths:
                my_abs = my + mouth_y0
                cv2.rectangle(face_color, (mx, my_abs), (mx + mw, my_abs + mh), (0, 60, 255), 2)
                cv2.putText(face_color, "mouth", (mx, my_abs - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 60, 255), 1)

    return frame, len(faces)


def is_fuck_off(lm):
    """Return True when only the middle finger is extended and hand is upright."""
    # Finger extended = tip.y < pip.y  (y roste dolů)
    # index=8/6, middle=12/10, ring=16/14, pinky=20/18
    index_up  = lm[8].y  < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up   = lm[16].y < lm[14].y
    pinky_up  = lm[20].y < lm[18].y
    # Ruka vzpřímená: zápěstí (0) musí být níž než základna prostředníčku (9)
    hand_upright = lm[0].y > lm[9].y
    return middle_up and not index_up and not ring_up and not pinky_up and hand_upright


def detect_hands(frame, hands_model, show_text=False):
    """Detect hand landmarks and draw a wired hand skeleton."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(rgb)

    n_hands = 0
    if results.multi_hand_landmarks:
        h, w = frame.shape[:2]
        for hand_lm in results.multi_hand_landmarks:
            n_hands += 1
            lm = hand_lm.landmark
            fuck_off = is_fuck_off(lm)

            # Draw connections (bones)
            lm_color = mp_draw.DrawingSpec(
                color=(0, 0, 255) if fuck_off else LANDMARK_COLOR.color,
                thickness=2, circle_radius=4
            )
            mp_draw.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS, lm_color, CONNECTION_COLOR,
            )

            # Gesture label + zvuk
            if fuck_off:
                if show_text:
                    wx = int(lm[0].x * w)
                    wy = int(lm[0].y * h)
                    cv2.putText(frame, "FUCK OFF!", (wx - 40, wy + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                global _last_fuck_off
                now = time.time()
                if now - _last_fuck_off >= FUCK_OFF_COOLDOWN:
                    _last_fuck_off = now
                    winsound.PlaySound(FUCK_OFF_SOUND,
                                       winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)

    return frame, n_hands


def find_cameras(max_index=5):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
        cap.release()
    return available


def select_camera():
    print("Searching for cameras...")
    cams = find_cameras()
    if not cams:
        print("[ERROR] No camera found.")
        sys.exit(1)

    if len(cams) == 1:
        print(f"Found 1 camera (index {cams[0]}).")
        return cams[0]

    print(f"Found cameras: {cams}")
    while True:
        try:
            choice = int(input(f"Select camera index {cams}: "))
            if choice in cams:
                return choice
        except ValueError:
            pass
        print("Invalid choice, try again.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", "-c", type=int, default=None,
                        help="Camera index (0, 1, ...). Auto-detected if not specified.")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available cameras and exit.")
    args = parser.parse_args()

    if args.list:
        cams = find_cameras()
        print(f"Available cameras: {cams}")
        sys.exit(0)

    face_cc, mouth_cc = load_cascades()
    glasses_img, glasses_mask = load_glasses()

    cam_index = args.camera if args.camera is not None else select_camera()
    cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_index}.")
        sys.exit(1)

    print(f"Camera {cam_index} started. Press Q to quit.")

    scr_w, scr_h = get_screen_size()

    win = "Face + Wired Hand Detector"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands_model, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh_model:

        show_text = False  # výchozí stav: texty skryté, O = přepnutí

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Empty frame, skipping.")
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mesh_results = face_mesh_model.process(rgb)

            frame, n_faces = detect_face_features(frame, face_cc, mouth_cc, mesh_results, glasses_img, glasses_mask, show_text)
            frame, n_hands = detect_hands(frame, hands_model, show_text)

            if show_text:
                cv2.putText(frame, f"Faces: {n_faces}  Hands: {n_hands}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "Q = quit  |  O = toggle info", (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            display = cv2.resize(frame, (scr_w, scr_h))
            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("o"):
                show_text = not show_text
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
