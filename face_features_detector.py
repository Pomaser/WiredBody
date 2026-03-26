import os
import sys


def resource_path(rel: str) -> str:
    """Vrátí absolutní cestu k souboru — funguje jak při vývoji, tak v PyInstaller .exe."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


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
import json
import logging
import math
import random
import time
import traceback
import winsound
import numpy as np
import cv2
import argparse
import mediapipe as mp

GLASSES_PATH      = resource_path("glasses_single.png")
JOINT_PATH        = resource_path("joint_single.png")
FUCK_OFF_SOUND    = resource_path("fuck_off.wav")
FUCK_OFF_COOLDOWN = 3.0   # sekundy mezi opakovaným přehráním

_last_fuck_off = 0.0
GLASSES_SCALE    = 0.75   # 1.0 = šířka obličeje
GLASSES_OFFSET_Y = 0
GLASSES_SMOOTH   = 0.75

JOINT_SCALE      = 0.4    # délka jointu jako násobek šířky obličeje
JOINT_SMOOTH     = 0.75

# Stav pro EMA vyhlazení
_g_smooth = {"gx": None, "gy": None, "angle": 0.0}
_j_smooth = {"jx": None, "jy": None}
_smoke: list = []
MAX_SMOKE = 80
SMOKE_SPAWN = 3   # částice za snímek


def _smoke_update_draw(frame, tip_x, tip_y):
    """Particle kouře u hořícího konce jointu."""
    h, w = frame.shape[:2]

    # Spawn nových částic
    for _ in range(SMOKE_SPAWN):
        if len(_smoke) < MAX_SMOKE:
            _smoke.append({
                "x": float(tip_x) + random.gauss(0, 2),
                "y": float(tip_y) + random.gauss(0, 2),
                "vx": random.gauss(0, 0.4),
                "vy": -random.uniform(0.6, 2.0),
                "size": random.uniform(2, 5),
                "alpha": random.uniform(0.25, 0.55),
                "age": 0,
                "max_age": random.randint(25, 55),
            })

    # Aktualizace fyziky + kreslení na jediný buffer
    smoke_buf = np.zeros((h, w), dtype=np.float32)
    alive = []
    for p in _smoke:
        p["age"] += 1
        if p["age"] >= p["max_age"]:
            continue
        p["x"]  += p["vx"]
        p["y"]  += p["vy"]
        p["vx"] += random.gauss(0, 0.18)   # turbulence
        p["size"] += 0.35                   # rozptyl
        life = 1.0 - p["age"] / p["max_age"]
        intensity = p["alpha"] * life
        cx, cy = int(p["x"]), int(p["y"])
        sz = max(1, int(p["size"]))
        y1, y2 = max(0, cy - sz), min(h, cy + sz + 1)
        x1, x2 = max(0, cx - sz), min(w, cx + sz + 1)
        if y2 > y1 and x2 > x1:
            cv2.circle(smoke_buf, (cx, cy), sz, float(intensity), -1)
        alive.append(p)
    _smoke[:] = alive

    if smoke_buf.max() < 0.01:
        return
    smoke_soft = cv2.GaussianBlur(smoke_buf, (31, 31), 0)
    smoke_soft = np.clip(smoke_soft, 0, 1)[..., np.newaxis]
    smoke_color = np.array([210, 210, 215], dtype=np.float32)
    frame[:] = (frame.astype(np.float32) * (1 - smoke_soft * 0.85)
                + smoke_color * smoke_soft * 0.85).astype(np.uint8)


def _ema(prev, cur, alpha):
    """Exponenciální klouzavý průměr."""
    return cur if prev is None else alpha * cur + (1.0 - alpha) * prev

def _load_png(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARNING] Cannot load: {path}")
        return None, None
    if img.shape[2] == 4:
        mask = img[:, :, 3]
        img  = img[:, :, :3]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    return img, mask

def load_glasses():
    return _load_png(GLASSES_PATH)

def load_joint():
    return _load_png(JOINT_PATH)


def overlay_image(bg, overlay, mask, x, y):
    oh, ow = overlay.shape[:2]
    bh, bw = bg.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + ow), min(bh, y + oh)
    if x2 <= x1 or y2 <= y1:
        return
    ov  = overlay[y1 - y:y2 - y, x1 - x:x2 - x].astype(float)
    a   = mask[y1 - y:y2 - y, x1 - x:x2 - x].astype(float) / 255.0
    roi = bg[y1:y2, x1:x2].astype(float)
    bg[y1:y2, x1:x2] = (ov * a[..., np.newaxis] + roi * (1.0 - a[..., np.newaxis])).astype(np.uint8)


def get_screen_size():
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

# ---------------------------------------------------------------------------
# Haar cascade classifiers (face, eyes, mouth)
# ---------------------------------------------------------------------------
def _cascade_path(filename: str) -> str:
    """Vrátí cestu ke cascade XML — z bundlu (exe) nebo z cv2.data (dev)."""
    bundled = resource_path(filename)
    if os.path.exists(bundled):
        return bundled
    return cv2.data.haarcascades + filename

FACE_CASCADE  = _cascade_path("haarcascade_frontalface_default.xml")
MOUTH_CASCADE = _cascade_path("haarcascade_smile.xml")

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

    # Pivot = bod mezi očima v souřadnicích obrázku brýlí
    eye_row = int(g_h * 0.35)   # shodné s raw_gy výpočtem výše
    pivot_x = g_w // 2
    pivot_y = eye_row

    rad = math.radians(abs(angle))
    pad_x = int(g_h * math.sin(rad) / 2) + 1
    pad_y = int(g_w * math.sin(rad) / 2) + 1
    g_img  = cv2.copyMakeBorder(g_img,  pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
    g_mask = cv2.copyMakeBorder(g_mask, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
    M = cv2.getRotationMatrix2D((pivot_x + pad_x, pivot_y + pad_y), angle, 1.0)
    g_img  = cv2.warpAffine(g_img,  M, (g_w + 2*pad_x, g_h + 2*pad_y))
    g_mask = cv2.warpAffine(g_mask, M, (g_w + 2*pad_x, g_h + 2*pad_y))

    overlay_image(frame, g_img, g_mask,
                  int(_g_smooth["gx"]) - pad_x,
                  int(_g_smooth["gy"]) - pad_y + eye_row - pivot_y)


def _draw_joint(frame, mesh_results, joint_img, joint_mask):
    """Overlay joint na střed úst; levý horní roh obrázku = střed úst."""
    if joint_img is None or joint_mask is None:
        return
    if mesh_results is not None and mesh_results.multi_face_landmarks:
        h, w = frame.shape[:2]
        lm = mesh_results.multi_face_landmarks[0].landmark
        # Střed úst: průměr levého (61) a pravého (291) koutku
        mx = int((lm[61].x + lm[291].x) / 2 * w)
        my = int((lm[61].y + lm[291].y) / 2 * h)
        _j_smooth["jx"] = _ema(_j_smooth["jx"], mx, JOINT_SMOOTH)
        _j_smooth["jy"] = _ema(_j_smooth["jy"], my, JOINT_SMOOTH)

    if _j_smooth["jx"] is None:
        return

    jx = int(_j_smooth["jx"])
    jy = int(_j_smooth["jy"])
    angle = _g_smooth["angle"]  # stejný náklon jako brýle

    # Velikost jointu odvozená od šířky obličeje z brýlí
    face_w = max(int(_g_smooth.get("gw", 60) / GLASSES_SCALE), 40)
    j_w = int(face_w * JOINT_SCALE)
    j_h = int(j_w * joint_img.shape[0] / joint_img.shape[1])
    j_img  = cv2.resize(joint_img,  (j_w, j_h))
    j_mask = cv2.resize(joint_mask, (j_w, j_h))

    # Padding proti ořezu rohů při rotaci
    rad = math.radians(abs(angle))
    pad_x = int(j_h * math.sin(rad) / 2) + 1
    pad_y = int(j_w * math.sin(rad) / 2) + 1
    j_img  = cv2.copyMakeBorder(j_img,  pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
    j_mask = cv2.copyMakeBorder(j_mask, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)

    # Pivot = levý horní roh originálu = (pad_x, pad_y) v paddovaném obrázku
    M = cv2.getRotationMatrix2D((pad_x, pad_y), angle, 1.0)
    j_img  = cv2.warpAffine(j_img,  M, (j_w + 2*pad_x, j_h + 2*pad_y))
    j_mask = cv2.warpAffine(j_mask, M, (j_w + 2*pad_x, j_h + 2*pad_y))

    # Levý horní roh originálu (= pad_x, pad_y v paddovaném) musí být na (jx, jy)
    overlay_image(frame, j_img, j_mask, jx - pad_x, jy - pad_y)

    # Kouř — hořící konec jointu = pravý dolní roh obrázku rotovaný do světa
    # OpenCV rot. konvence: x' = cos*x + sin*y,  y' = -sin*x + cos*y
    a_rad = math.radians(angle)
    tip_x = int(jx + j_w * math.cos(a_rad) + j_h * math.sin(a_rad))
    tip_y = int(jy - j_w * math.sin(a_rad) + j_h * math.cos(a_rad))
    _smoke_update_draw(frame, tip_x, tip_y)


_cached_faces = []


def detect_face_features(frame, face_cc, mouth_cc, mesh_results=None, glasses_img=None, glasses_mask=None,
                         joint_img=None, joint_mask=None, show_text=False, run_haar=True):
    global _cached_faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Brýle + joint řídí Face Mesh — nezávislé na Haar
    _draw_glasses(frame, mesh_results, glasses_img, glasses_mask)
    _draw_joint(frame, mesh_results, joint_img, joint_mask)

    if run_haar:
        detected = face_cc.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        _cached_faces = detected if len(detected) else []
    faces = _cached_faces

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


def load_config() -> dict:
    # 1) vedle .exe / vedle .py — uživatel může editovat
    exe_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    external = os.path.join(exe_dir, "config.json")
    # 2) uvnitř bundlu — výchozí hodnoty
    bundled = resource_path("config.json")
    path = external if os.path.exists(external) else bundled
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


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
    joint_img,   joint_mask   = load_joint()

    if args.camera is not None:
        cam_index = args.camera
    else:
        cfg = load_config()
        cam_cfg = cfg.get("camera", {})
        preferred = cam_cfg.get("preferred", None)
        fallback = cam_cfg.get("fallback", [])
        available = find_cameras()
        if preferred is not None and preferred in available:
            cam_index = preferred
        else:
            if preferred is not None:
                print(f"[WARN] Preferred camera {preferred} not found.")
            for fb in fallback:
                if fb in available:
                    cam_index = fb
                    print(f"Using fallback camera {fb}.")
                    break
            else:
                cam_index = select_camera()

    print(f"Opening camera {cam_index}...")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_index}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"Camera {cam_index} started. Press Q to quit.")

    scr_w, scr_h = get_screen_size()

    win = "Face + Wired Hand Detector"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow(win, 0, 0)
    cv2.resizeWindow(win, scr_w, scr_h)
    fullscreen = False

    PROC_SCALE   = 0.5   # detekce probíhá v polovičním rozlišení
    HAAR_EVERY_N = 3     # Haar kaskáda každý N-tý snímek

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

        show_text = False
        frame_idx = 0
        _last_faces = 0
        _last_hands = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Empty frame, skipping.")
                continue

            frame_idx += 1
            h0, w0 = frame.shape[:2]

            # Zmenšení pro detekci
            small = cv2.resize(frame, (int(w0 * PROC_SCALE), int(h0 * PROC_SCALE)),
                               interpolation=cv2.INTER_LINEAR)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mesh_results = face_mesh_model.process(rgb_small)

            # Haar jen každý N-tý snímek, jinak přeskoč (EMA udržuje polohu)
            run_haar = (frame_idx % HAAR_EVERY_N == 0)
            small, n_faces = detect_face_features(
                small, face_cc, mouth_cc, mesh_results,
                glasses_img, glasses_mask, joint_img, joint_mask,
                show_text, run_haar=run_haar
            )
            small, n_hands = detect_hands(small, hands_model, show_text)

            # Výsledek zpět na původní rozlišení
            frame = cv2.resize(small, (w0, h0), interpolation=cv2.INTER_LINEAR)
            _last_faces, _last_hands = n_faces, n_hands

            if show_text:
                cv2.putText(frame, f"Faces: {_last_faces}  Hands: {_last_hands}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "Q = quit  |  O = toggle info", (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            scale = min(scr_w / frame.shape[1], scr_h / frame.shape[0])
            new_w = int(frame.shape[1] * scale)
            new_h = int(frame.shape[0] * scale)
            resized = cv2.resize(frame, (new_w, new_h))
            display = np.zeros((scr_h, scr_w, 3), dtype=np.uint8)
            x0 = (scr_w - new_w) // 2
            y0 = (scr_h - new_h) // 2
            display[y0:y0+new_h, x0:x0+new_w] = resized
            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("o"):
                show_text = not show_text
            if key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    win, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                )
            if frame_idx > 10 and cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__)),
        "facedetector.log"
    )
    logging.basicConfig(
        filename=log_path, filemode="w", level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info("Starting FaceDetector")
    try:
        main()
    except Exception:
        logging.critical("Unhandled exception:\n" + traceback.format_exc())
        raise
    logging.info("FaceDetector exited normally")
