import os
import frame_extractor
import dataset


def extract_png_frames(video_directory):
    total = 0
    for root, _, files in os.walk(video_directory):
        for file_name in files:
            if file_name.endswith(".mp4"):
                total += 1

    print(f"Found {total} videos in '{video_directory}'.")

    count = 1
    for root, _, files in os.walk(video_directory):
        for file_name in files:
            if file_name.endswith(".mp4"):
                video_file = os.path.join(root, file_name)
                print(f"Processing: {video_file} ({count}/{total})")
                frame_extractor.extract_frames(video_file)
                count += 1


if __name__ == "__main__":
    # extract_png_frames(os.path.join("datasets", "GITW_light"))
    dataset.create_dataset("datasets", train=True)
