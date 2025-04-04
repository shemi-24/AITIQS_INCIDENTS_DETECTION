import face_recognition
import os
import pickle

# Define dataset path
known_faces_dir = "/Users/zoftcares/StudioProjects/fall_detection_system/dataset/faces"

face_encodings = []
face_names = []

# Loop through each person directory
for person_name in os.listdir(known_faces_dir):
    person_path = os.path.join(known_faces_dir, person_name)
    
    if not os.path.isdir(person_path):  # Skip non-folder files
        continue
    
    # Loop through images in each person's folder
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        
        try:
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:  # If encoding is found
                face_encodings.append(encodings[0])
                face_names.append(person_name)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

# Create model directory if not exists
os.makedirs("model", exist_ok=True)

# Save encodings to a file
with open("model/face_encodings.pkl", "wb") as f:
    # pickle.dump((face_encodings, face_names), f)
    pickle.dump({"encodings": face_encodings, "names": face_names}, f)

print(f"Face encodings saved! Total encodings: {len(face_encodings)}")
