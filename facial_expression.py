import cv2
from deepface import DeepFace


def load_emotion_detector():
  """Loads the face cascade classifier for emotion detection."""
  return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def capture_video():
  """Starts capturing video from the default webcam."""
  return cv2.VideoCapture(0)


def analyze_emotion(image):
  """Analyzes the dominant emotion in the provided image.

  Args:
      image: The image to analyze (BGR format).

  Returns:
      The dominant emotion detected in the image, or None if no face is found.
  """
  result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
  if result:
    return result[0]['dominant_emotion']
  else:
    return None


def display_emotions(frame, detected_faces, emotions):
  """Displays rectangles and labels for detected emotions on the frame.

  Args:
      frame: The frame to display (BGR format).
      detected_faces: A list of tuples representing detected faces (x, y, w, h).
      emotions: A list of emotions corresponding to the detected faces.
  """
  for (x, y, w, h), emotion in zip(detected_faces, emotions):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


def main():
  """Main function to perform real-time emotion detection."""

  emotion_detector = load_emotion_detector()
  cap = capture_video()

  while True:
    ret, frame = cap.read()

    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detected_faces = emotion_detector.detectMultiScale(
        grayscale_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    emotions = []
    for face_region in [frame[y:y + h, x:x + w] for (x, y, w, h) in detected_faces]:
      emotion = analyze_emotion(face_region)
      if emotion:
        emotions.append(emotion)
      else:
        emotions.append("No Face Detected")

    display_emotions(frame, detected_faces, emotions)

    cv2.imshow('Real-time Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
