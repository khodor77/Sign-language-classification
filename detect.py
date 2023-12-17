import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model_path = "sign.h5"  # Replace with the actual path to your trained model
model = load_model(model_path)

# Compile the model with your desired settings
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Create a mapping for class labels
class_mapping = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}

# Open webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to match the input size expected by your model
    frame_resized = cv2.resize(frame, (100, 100))

    # Expand dimensions to create a batch-sized image
    img_array = np.expand_dims(frame_resized, axis=0)

    # Modify the preprocessing based on your model's requirements
    img_array = img_array.astype(np.float32) / 255.0

    # Make predictions using your model
    predictions = model.predict(img_array)

    # Assuming a single-class prediction
    predicted_class = np.argmax(predictions)
    predicted_label = class_mapping[predicted_class]

    # Display the predicted label on the frame
    cv2.putText(
        frame,
        predicted_label,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Display the frame
    cv2.imshow("Model Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
