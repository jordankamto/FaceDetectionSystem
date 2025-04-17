import face_recognition
import cv2
import numpy as np


video_capture = cv2.VideoCapture(0)


try:
    AUDREY_image = face_recognition.load_image_file("audrey.jpg")
    AUDREY_face_encoding = face_recognition.face_encodings(AUDREY_image)[0]
except Exception as e:
    print(f"Error processing Audrey's image: {e}")
    exit(1)


try:
    CHARLES_image = face_recognition.load_image_file("charles test.jpg")
    CHARLES_face_encoding = face_recognition.face_encodings(CHARLES_image)[0]
except Exception as e:
    print(f"Error processing Charles's image: {e}")
    exit(1)
try:
    DORCHELLE_image = face_recognition.load_image_file("dorchelle test.JPG")
    DORCHELLE_face_encoding = face_recognition.face_encodings(DORCHELLE_image)[0]
except Exception as e:
    print(f"Error processing Dorchelle's image: {e}")
    exit(1)


known_face_encodings = [
    AUDREY_face_encoding,
    CHARLES_face_encoding,
    DORCHELLE_face_encoding
]
known_face_names = [
    "Audrey",
    "Charles",
    "Dorchelle",
]

while True:
    
    ret, frame = video_capture.read()

   
    rgb_frame = frame[:, :, ::-1]

    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Inconnu"

       

       
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

       
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

       
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

   
    cv2.imshow('Video', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
