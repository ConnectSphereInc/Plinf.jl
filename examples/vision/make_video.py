import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def combine_mp4_files(directory):
    # List all files in the directory
    files = [f for f in os.listdir(directory) if f.startswith('infer_') and f.endswith('.mp4')]
    files.sort(key=natural_sort_key)

    # Create a list to store VideoFileClip objects
    clips = []
    
    for t, file in enumerate(files):
        file_path = os.path.join(directory, file)
        clip = VideoFileClip(file_path)
        # Calculate the midpoint of the video duration
        if t != 0:
            midpoint = clip.duration / 2
        # Trim the clip to only include the second half
            clip = clip.subclip(midpoint, clip.duration)
        clips.append(clip)
        print(f"Processed {file}.")

    if clips:
        # Concatenate all video clips
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Output the final combined video
        output_path = os.path.join(directory, "combined_output.mp4")
        final_clip.write_videofile(output_path, codec='libx264')
        print(f"Combined video saved to {output_path}")
    else:
        print("No video files found to process.")

# Example usage
combine_mp4_files("examples/vision/output/")
