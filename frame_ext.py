import cv2
import os

def extract_frames(video_path, output_folder, video_name, frame_rate=30):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            filename = os.path.join(output_folder, f"{video_name}_frame_{saved:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
        count += 1
    cap.release()

# Ana klasörleri oluştur
os.makedirs("videos", exist_ok=True)
os.makedirs("frames", exist_ok=True)

# videos klasöründeki tüm videoları işle
video_folder = "videos"
for video_file in os.listdir(video_folder):
    if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]  # Video adını al (uzantısız)
        print(f"İşleniyor: {video_file}")
        extract_frames(video_path, "frames", video_name, frame_rate=30)  # 30 FPS video için saniyede 1 frame
        print(f"Tamamlandı: {video_file}")
