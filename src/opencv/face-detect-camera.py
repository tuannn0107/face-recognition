import dlib
import cv2


video_capture = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
while True:
    ret, frame = video_capture.read()
    if ret == False:
        break
    frame = cv2.resize(frame, (750, 750), fx=0.5, fy=0.5)  # resize frame (optional)
    frame = frame[:, :, 0:3]
    detected_faces = face_detector(frame, 1)

    print("Found {} faces in the frame.".format(len(detected_faces)))
    for i, face_rect in enumerate(detected_faces):
        cv2.rectangle(frame, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
