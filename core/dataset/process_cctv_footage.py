import sys 
import os 
import cv2
import subprocess

# Get absolute path appended tot sys path of project
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from insightface.app import FaceAnalysis
from core.detection.detect import detect_faces, align_face

def get_all_name(dir):
    '''
    returns all the unique persons name in the dir
    '''
    files = os.listdir(dir)
    person_list = set()
    for file in files:
        name = file.split('_')[0].strip().lower()
        person_list.add(name) 
    
    return person_list

def repair_video_with_ffmpeg(video_path):
    print("⚠️ Video decoding failed. Repairing video with FFmpeg...")

    tmp_path = video_path + ".tmp.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-fflags", "+genpts",
        "-err_detect", "ignore_err",
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-an",
        tmp_path
    ]

    subprocess.run(cmd, check=True)

    # Replace original video
    os.replace(tmp_path, video_path)

    print("✅ Video repaired successfully.")


def extract_frames_with_repair(video_path, output_folder, video_number = 1, save_freq = 10):
    """
    Extracts all frames from a video file and saves them as JPEG images.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory to save the extracted frames.
    """

    def run_extraction():
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return 0

        frame_count = 0
        print(f"Starting frame extraction from {video_path}...")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_filename = os.path.join(
                output_folder,
                f"frame_{video_number}_{frame_count:04d}.jpg"
            )

            if frame_count % save_freq == 0:
                cv2.imwrite(frame_filename, frame)

            frame_count += 1

        cap.release()
        print(f"Video capture object released. Total frames: {frame_count}")
        return frame_count


    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # ---------- First attempt ----------
    frame_count = run_extraction()

    # ---------- If failed, repair video ----------
    if frame_count == 0:
        print("⚠️ No frames extracted. Attempting FFmpeg repair...")

        tmp_video = video_path + ".tmp.mp4"

        cmd = [
            "ffmpeg",
            "-y",
            "-fflags", "+genpts",
            "-err_detect", "ignore_err",
            "-i", video_path,
            "-c:v", "copy",
            "-an",
            tmp_video
        ]

        try:
            subprocess.run(cmd, check=True)
            os.replace(tmp_video, video_path)
            print("✅ Video repaired. Retrying extraction...")

            # Retry extraction
            run_extraction()

        except Exception as e:
            print("❌ FFmpeg repair failed:", e)

def extract_frames(video_path, output_folder, video_number = 1, save_freq = 10):
    """
    Extracts all frames from a video file and saves them as JPEG images.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Directory to save the extracted frames.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    print(f"Starting frame extraction from {video_path}...")

    while True:
        # Read a new frame
        ret, frame = cap.read()
        
        # Break the loop if the video ends or no frame is read
        if not ret:
            break
        
        # Define the filename for the current frame (e.g., frame_0000.jpg)
        frame_filename = os.path.join(output_folder, f"frame_{video_number}_{frame_count:04d}.jpg")
        
        if frame_count%save_freq==0:
            # Save the frame as an image file
            cv2.imwrite(frame_filename, frame)    
            # print(f"Frame {frame_count} saved as {frame_filename}")

        frame_count += 1

    # Release the video capture object and free resources
    cap.release()
    print(f"Video capture object released. Total frames: {frame_count}")


def old_main():
    app = FaceAnalysis(name='buffalo_l',
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                   allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)


    cctv_footage_dir = '/media/ayon/New Volume/Hamim_FR/office_vdos'
    output_dir = '/media/ayon/New Volume/Hamim_FR/ha_meem_ai_surveillance/dataset/office_dataset'
    aligned_face_dir = '/media/ayon/New Volume/Hamim_FR/ha_meem_ai_surveillance/dataset/office_dataset_aligned'

    all_videos = os.listdir(cctv_footage_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(aligned_face_dir, exist_ok=True)

    names = get_all_name(cctv_footage_dir)
    print(f"Found Names: {names}")
    for name in names:
        print(f"processing dataset for for {name}")
        filtered_vdos = [video for video in all_videos if name in video.lower()]
        save_dir = os.path.join(output_dir, name)
        aligned_save_dir = os.path.join(aligned_face_dir, name)
        os.makedirs(aligned_save_dir, exist_ok=True)

        for i, vdo_name in enumerate(filtered_vdos):
            extract_frames_with_repair(os.path.join(cctv_footage_dir, vdo_name), output_folder=save_dir, video_number=i, save_freq=5)
        
        for image in os.listdir(save_dir):
            img_path = os.path.join(save_dir, image)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read {img_path}")
                continue
            
            #for training only 1 face
            faces = detect_faces(img, app)
            if faces is None:
                continue
            face = faces[0]
            face_width_px = face['bbox'][2] - face['bbox'][0]
            if face_width_px<150: #filter based on face width
                continue
            
            aligned_face = align_face(img, face.kps, output_size=(112,112)) #adaface expects this size
            
            out_path = os.path.join(aligned_save_dir, image)
            cv2.imwrite(out_path, aligned_face)
            
            
def process_aligned_faces(face_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_faces = os.listdir(face_dir)
    print(f"No total faces found {len(all_faces)}")

    names = set()
    for faces in all_faces:
         name = faces.lower().split('_')[0]
         names.add(name)
    
    print(f"Different Unique names found {names}")

    for name in names:
        name_dir = os.path.join(output_dir, name)
        os.makedirs(name_dir, exist_ok=True)

        filtered_faces = [face for face in all_faces if face.lower().split('_')[0]==name]
        print(f"total faces found for {name} is {len(filtered_faces)}")

        
def main():
    app = FaceAnalysis(name='buffalo_l',
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                   allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)


    save_dir = '/media/ayon/New Volume/Hamim_FR/ha_meem_ai_surveillance/dataset/output'
    aligned_face_dir = '/media/ayon/New Volume/Hamim_FR/ha_meem_ai_surveillance/dataset/office_dataset_aligned'

    os.makedirs(aligned_face_dir, exist_ok=True)
    
    for i, image in enumerate(os.listdir(save_dir)):
        print(f"processing {i+1}/{len(os.listdir(save_dir))}", end='\r')
        img_path = os.path.join(save_dir, image)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}")
            continue
        
        #for training only 1 face
        faces = detect_faces(img, app)
        if faces is None:
            continue
        face = faces[0]
        face_width_px = face['bbox'][2] - face['bbox'][0]
        if face_width_px<150: #filter based on face width
            continue
        
        aligned_face = align_face(img, face.kps, output_size=(112,112)) #adaface expects this size
        
        out_path = os.path.join(aligned_face_dir, image)
        cv2.imwrite(out_path, aligned_face)

def main_new():
    app = FaceAnalysis(name='buffalo_l',
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                   allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)


    dataset_dir = '/media/ayon/New Volume/Hamim_FR/all_pictures'
    aligned_face_dir = '/media/ayon/New Volume/Hamim_FR/ha_meem_ai_surveillance/dataset/all_pic_aligned'
    os.makedirs(aligned_face_dir, exist_ok=True)

    names = os.listdir(dataset_dir)
    print(f"Found all persons = {names}")

    for name in names:
        if 'authorized' in name:
            continue

        name_data_dir = os.path.join(dataset_dir, name)
        name_save_dir = os.path.join(aligned_face_dir, name)
        os.makedirs(name_save_dir, exist_ok=True)

        for i, image in enumerate(os.listdir(name_data_dir)):
            img_path = os.path.join(name_data_dir, image)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read {img_path}")
                continue
            
            #for training only 1 face
            faces = detect_faces(img, app)
            if faces is None:
                continue
            face = faces[0]
            face_width_px = face['bbox'][2] - face['bbox'][0]
            if face_width_px<150: #filter based on face width
                continue
            
            aligned_face = align_face(img, face.kps, output_size=(112,112)) #adaface expects this size
            
            out_path = os.path.join(name_save_dir, image)
            cv2.imwrite(out_path, aligned_face)

        
        print(f"Found {len(os.listdir(name_save_dir))} eligible faces for {name}")

    

if __name__=="__main__":
    main_new()
    