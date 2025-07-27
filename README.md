Monocular SLAM System

This project implements a basic monocular SLAM system using OpenCV and Python, including a live webcam tracker and a Pygame 2D map viewer. It features ORB feature extraction and matching, essential matrix estimation and pose recovery, 3D landmark triangulation, session saving/loading with pickle, and a visual map viewer using Pygame.

Requirements:

- Python 3.7 or higher
- Required Python packages: opencv-python, numpy, pygame, scikit-learn, scipy
You can install them by running: pip install opencv-python numpy pygame scikit-learn scipy

To run the SLAM system on a live webcam feed, use:
python Slam.py --live
This processes frames from your default webcam and builds the SLAM map in real time.

To load a previously saved SLAM session, use:
python Slam.py --load path/to/slamsession.pkl

Optional command-line arguments include:
--export_dir: Directory where SLAM outputs (map, graph, logs) are saved. Default is "slam_output".
--frames: Maximum number of frames to process. Default is 500.

Example usage:
python Slam.py --live --frames 300 --export_dir my_slam_output

Outputs include a SLAM session pickle file named slam_session.pkl, a 3D map saved as a .ply point cloud file, a pose graph saved as a .g2o file, and a log file with session info in the export directory.

In the Pygame viewer, use the arrow keys to pan the map, the "+" or "=" key to zoom in, the "-" key to zoom out, and ESC or closing the window to exit.

Make sure your webcam is connected and accessible by OpenCV. This project currently uses a simple ORB feature matcher without loop closure optimization. The Pygame viewer visualizes camera poses as white dots and triangulated landmarks as red squares.

If you encounter errors related to missing packages, check your Python environment and reinstall dependencies. For camera access issues, verify system permissions and camera availability.

Feel free to open issues or contribute improvements!
