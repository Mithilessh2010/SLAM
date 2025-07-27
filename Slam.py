import cv2
import numpy as np
import os
import pickle
import pygame
import datetime
import argparse
from collections import deque
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser(description="Monocular SLAM System (No GTSAM)")
parser.add_argument('--load', type=str, help='Path to saved SLAM session')
parser.add_argument('--live', action='store_true', help='Run live SLAM from webcam')
parser.add_argument('--export_dir', type=str, default='slam_output', help='Directory to export SLAM outputs')
parser.add_argument('--frames', type=int, default=500, help='Max number of frames to process')
args = parser.parse_args()

EXPORT_DIR = args.export_dir
MAX_FRAMES = args.frames
FEATURES_PER_FRAME = 2000
LOOP_THRESHOLD = 0.75
SESSION_PATH = "slam_session.pkl"

def serialize_keypoints(kp_list):
    return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp_list]

if args.load:
    print("[Session] Loading SLAM session from", args.load)
    with open(args.load, 'rb') as f:
        data = pickle.load(f)
    poses = data["poses"]
    keyframes = data["keyframes"]
    landmarks = data["landmarks"]
    print("[Session] Loaded", len(poses), "poses and", len(landmarks), "landmarks")
    exit(0)

orb = cv2.ORB_create(FEATURES_PER_FRAME)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Webcam not detected")

def compose_pose(T1, T2):
    return T1 @ T2

poses = [np.eye(4)]
keyframes = []
landmarks = []
bow_db = []
frame_id = 0
loop_queue = deque(maxlen=50)

prev_kp, prev_desc, prev_img = None, None, None

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
export_subdir = os.path.join(EXPORT_DIR, timestamp)
os.makedirs(export_subdir, exist_ok=True)
log_path = os.path.join(export_subdir, "slam_log.txt")
with open(log_path, "w") as log:
    log.write(f"SLAM Session Started: {datetime.datetime.now()}\n")
    log.write(f"Parameters: MAX_FRAMES={MAX_FRAMES}, FEATURES={FEATURES_PER_FRAME}\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, desc = orb.detectAndCompute(gray, None)

    if prev_desc is not None and desc is not None:
        matches = bf.match(prev_desc, desc)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > 30:
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
            E, mask = cv2.findEssentialMat(pts2, pts1, focal=1.0, pp=(0., 0.), method=cv2.RANSAC)
            if E is not None:
                _, R_mat, t, mask_pose = cv2.recoverPose(E, pts2, pts1)
                T_rel = np.eye(4)
                T_rel[:3, :3] = R_mat
                T_rel[:3, 3] = t.flatten()
                T_curr = compose_pose(poses[-1], T_rel)
                poses.append(T_curr)

                def bow_desc(d): return np.mean(d.astype(np.float32), axis=0)
                if len(bow_db) > 5:
                    current_bow = bow_desc(desc)
                    nbrs = NearestNeighbors(n_neighbors=1).fit(np.array(bow_db))
                    dist, idx = nbrs.kneighbors([current_bow])
                    if dist[0][0] < LOOP_THRESHOLD:
                        pass  # No loop closure correction here

                def triangulate(k1, k2, m, pose1, pose2):
                    proj1 = np.hstack((pose1[:3, :3], pose1[:3, 3:4]))
                    proj2 = np.hstack((pose2[:3, :3], pose2[:3, 3:4]))
                    pts1 = np.float32([k1[m.queryIdx].pt for m in m])
                    pts2 = np.float32([k2[m.trainIdx].pt for m in m])
                    pts4d = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
                    pts4d /= pts4d[3]
                    return pts4d[:3].T[np.isfinite(pts4d).all(axis=0)]

                if frame_id > 1:
                    points = triangulate(prev_kp, kp, matches, poses[-2], poses[-1])
                    landmarks.extend(points)

                serialized_kp = serialize_keypoints(kp)
                keyframes.append((frame_id, T_curr, serialized_kp, desc))
                bow_db.append(bow_desc(desc))
                vis = cv2.drawMatches(prev_img, prev_kp, frame, kp, matches[:20], None)
                cv2.imshow("Tracking", vis)

    prev_kp, prev_desc, prev_img = kp, desc, gray
    frame_id += 1
    if frame_id > MAX_FRAMES or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("[Save] Writing SLAM session...")
with open(SESSION_PATH, "wb") as f:
    pickle.dump({"poses": poses, "keyframes": keyframes, "landmarks": landmarks}, f)

def save_ply(points, path):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(points)))
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
save_ply(landmarks, os.path.join(export_subdir, "map.ply"))

def save_g2o(path, poses):
    with open(path, "w") as f:
        for i, T in enumerate(poses):
            t = T[:3, 3]
            f.write(f"VERTEX_SE3:QUAT {i} {t[0]} {t[1]} {t[2]} 0 0 0 1\n")
        for i in range(1, len(poses)):
            f.write(f"EDGE_SE3:QUAT {i-1} {i} 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1\n")
save_g2o(os.path.join(export_subdir, "graph.g2o"), poses)

# --- GUI Viewer ---
pygame.init()
W, H = 800, 800
win = pygame.display.set_mode((W, H))
pygame.display.set_caption("SLAM Map Viewer")
WHITE, RED = (255, 255, 255), (255, 0, 0)
running = True
scale = 40
camera_offset = [0, 0]
zoom = 1.0

def world_to_screen(x, y):
    try:
        sx = int(W / 2 + float(x + camera_offset[0]) * scale * zoom)
        sy = int(H / 2 - float(y + camera_offset[1]) * scale * zoom)
        return (sx, sy)
    except (ValueError, TypeError, OverflowError):
        return (0, 0)

while running:
    win.fill((20, 20, 20))
    for T in poses:
        x, y = T[0, 3], T[1, 3]
        screen_pt = world_to_screen(x, y)
        pygame.draw.circle(win, WHITE, screen_pt, 2)

    for pt in landmarks:
        x, y = pt[0], pt[1]
        screen_pt = world_to_screen(x, y)
        if isinstance(screen_pt, tuple) and len(screen_pt) == 2:
            sx, sy = screen_pt
            try:
                sx, sy = int(sx), int(sy)
                if 0 <= sx < W and 0 <= sy < H:
                    # Draw a 2x2 pixel rectangle for landmarks safely
                    pygame.draw.rect(win, RED, (sx, sy, 2, 2))
            except Exception:
                pass  # Ignore invalid points

    pygame.display.flip()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]: camera_offset[0] += 0.1
    if keys[pygame.K_RIGHT]: camera_offset[0] -= 0.1
    if keys[pygame.K_UP]: camera_offset[1] -= 0.1
    if keys[pygame.K_DOWN]: camera_offset[1] += 0.1
    if keys[pygame.K_EQUALS]: zoom *= 1.05
    if keys[pygame.K_MINUS]: zoom *= 0.95

    for event in pygame.event.get():
        if event.type == pygame.QUIT or keys[pygame.K_ESCAPE]:
            running = False

pygame.quit()
