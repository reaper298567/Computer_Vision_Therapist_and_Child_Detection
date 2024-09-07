# Computer Vision Therapist and Child Detection and Tracking

This project detects and tracks therapists and children in a video using YOLOv5 and the SORT algorithm. The classification is done based on bounding box sizes, and the tracking ensures consistency of object identification over time.

## Project Structure

- `main.py`: Main script that performs object detection and tracking.
- `sort.py`: SORT algorithm for tracking.
- `requirements.txt`: Required libraries for running the project.
- `videos/`: Folder containing input videos.

## Running the Project

## Instructions:

  1. Place your video file in the videos/ folder.
  2. Install the required libraries using pip install -r requirements.txt.
  3. Run the main.py script to process the video and generate the output.