import cv2
import os


def extract_frames(video_path):
    if not os.path.exists(video_path):
        print(f"The file '{video_path}' doesn't exist.")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = "Frames"
    output_dir_path = os.path.join(os.path.dirname(video_path), output_dir)

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    else:
        for file in os.listdir(output_dir_path):
            os.remove(os.path.join(output_dir_path, file))

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Impossible to load the video '{video_path}'.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_dir_path, f"frame_{frame_count:05d}.png")
        print(f"\rExtracting frame {frame_count}...", end="")

        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print(f"\nExtracted {frame_count} frames from '{video_name}'.")
