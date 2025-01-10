import os
import frame_extractor


def extract_png_frames(video_directory):
    for root, _, files in os.walk(video_directory):
        for file_name in files:
            if file_name.endswith(".mp4"):
                video_file = os.path.join(root, file_name)
                print(f"Processing: {video_file}")
                frame_extractor.extract_frames(video_file)


if __name__ == "__main__":
    extract_png_frames(os.path.join("datasets", "GITW_light"))
