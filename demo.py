import math
import cv2
import numpy as np
import requests
import io
from PIL import Image
import matplotlib.cm as cm

def get_tile_xy(lat, lon, z):
    lat_rad = math.radians(lat)
    n = 2.0 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y

# --- Configuration for Vadodara City Segment ---
# Zoom level 14 is ideal for city-wide scale (~7km span). 
# Level 18 on this large area would download thousands of tiles and crash.
z = 14 

# Converted from DMS to Decimal Degrees.
# Note: Flipped the latitudes from the prompt so 'top' is actually the northernmost coordinate.
lat_top_left, lon_top_left = 22.318875, 73.165219      # 22°19'07.95"N, 73°09'54.79"E
lat_bottom_right, lon_bottom_right = 22.252569, 73.208156  # 22°15'09.25"N, 73°12'29.36"E

x1, y1 = get_tile_xy(lat_top_left, lon_top_left, z)
x2, y2 = get_tile_xy(lat_bottom_right, lon_bottom_right, z)

x_min, x_max = min(x1, x2), max(x1, x2)
y_min, y_max = min(y1, y2), max(y1, y2)

print("Step 1/8: Fetching satellite tiles...")
headers = {"user-agent": "Mozilla/5.0"}
rows = []

for y in range(y_min, y_max + 1):
    row_images = []
    for x in range(x_min, x_max + 1):
        url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
        response = requests.get(url, headers=headers)
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        row_images.append(img_bgr)
    row_stitched = np.hstack(row_images)
    rows.append(row_stitched)

satellite_img = np.vstack(rows)
satellite_height, satellite_width = satellite_img.shape[:2]
print(f"       Stitched satellite image size: {satellite_width}x{satellite_height}")

print("Step 2/8: Loading input image...")
# Remember to update this path to your new input image for Vadodara!
input_path = r"C:\Users\lenovo\Desktop\BISAG DEMO\Screenshot 2026-05-22 141321.png"
input_img = cv2.imread(input_path)
if input_img is None:
    raise FileNotFoundError(f"Input image not found at: {input_path}")
input_height, input_width = input_img.shape[:2]
print(f"       Input image size: {input_width}x{input_height}")

print("Step 3/8: Initializing SIFT and detecting features...")
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(input_img, None)
kp2, des2 = sift.detectAndCompute(satellite_img, None)
print(f"       Detected {len(kp1)} keypoints in input image.")
print(f"       Detected {len(kp2)} keypoints in satellite image.")

print("Step 4/8: Matching features with BFMatcher...")
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)
print(f"       Found {len(matches)} potential matches.")

print("Step 5/8: Applying Lowe's ratio test to filter matches...")
good_matches = []
ratio_threshold = 0.75
for m, n in matches:
    if m.distance < ratio_threshold * n.distance:
        good_matches.append(m)
print(f"       Matches after Lowe's test: {len(good_matches)}")

print("Step 6/8: Running RANSAC to remove geometric outliers...")
if len(good_matches) < 4:
    raise ValueError("Not enough filtered matches to compute spatial homography matrix via RANSAC.")

src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matches_mask = mask.ravel().tolist()

inliers = []
for i in range(len(good_matches)):
    if matches_mask[i] == 1:
        inliers.append(good_matches[i])
print(f"       Robust matches after RANSAC: {len(inliers)}")

print("Step 7/8: Drawing visualized inliers on output images...")
num_to_draw = min(20, len(inliers))  
circle_radius = 6              
colors = cm.hsv(np.linspace(0, 1, num_to_draw))

colors_bgr = []
for c in colors:
    b = int(c[2] * 255)
    g = int(c[1] * 255)
    r = int(c[0] * 255)
    colors_bgr.append((b, g, r))

out_img = input_img.copy()
out_sat = satellite_img.copy()

for i in range(num_to_draw):
    match = inliers[i]
    pt1 = tuple(np.round(kp1[match.queryIdx].pt).astype(int))
    pt2 = tuple(np.round(kp2[match.trainIdx].pt).astype(int))
    color = colors_bgr[i]
    
    cv2.circle(out_img, pt1, circle_radius, color, -1)
    cv2.circle(out_sat, pt2, circle_radius, color, -1)

print("Step 8/8: Saving output files...")
# Saving directly to your specified Desktop folder
cv2.imwrite(r"C:\Users\lenovo\Desktop\BISAG DEMO\output_image_highlighted.jpg", out_img)
cv2.imwrite(r"C:\Users\lenovo\Desktop\BISAG DEMO\output_satellite_highlighted.jpg", out_sat)
print("Demonstration successfully completed. Check your folder at C:\\Users\\lenovo\\Desktop\\BISAG DEMO")