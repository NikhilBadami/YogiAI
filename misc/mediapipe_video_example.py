import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
if __name__ == "__main__":
    # For static images:
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    path = "path/to/test/image"
    image = cv2.imread(path)
    print(image)
    image_hight, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw pose landmarks on the image.
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
      annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pose.close()
    """
    
    # For webcam input:
    pose = mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    
      # Flip the image horizontally for a later selfie-view display, and convert
      # the BGR image to RGB.
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      results = pose.process(image)
    
      # Draw the pose annotation on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      mp_drawing.draw_landmarks(
          image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
      cv2.imshow('MediaPipe Pose', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
    pose.close()
    cap.release()
    """
