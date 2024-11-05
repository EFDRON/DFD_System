import streamlit as st

st.set_page_config(layout="wide")


def intro():
    import streamlit as st
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image("Images/logo-removebg-preview.png")


    st.write("<div style='text-align: center; font-size:37px;'><strong>Driver Drowsiness Detection System</strong></div>", unsafe_allow_html=True)
    st.write("<div style='text-align: center; font-size:24px;'>Stay Awake, Stay Alive: Your Safety Matters!!!</div>", unsafe_allow_html=True)
    col1,col2,col3=st.columns(3)
    with col2:

        st.image("Images/Car.png")

    st.success("Select model on the side bar")
    st.sidebar.success("Select a model above.")
def YOLO11_cls():

    import numpy as np
    import cv2
    import mediapipe as mp
    import streamlit as st

    from ultralytics import YOLO
    import math
    import utils
    # Load YOLO models


    model = YOLO("Models_Final/YOLO11_eye.pt")
    model2 = YOLO("Models_Final/YOLO11_yawn.pt")
    st.markdown("""
           <style>
               
           </style>
           """, unsafe_allow_html=True)
    # MediaPipe setup
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    FONTS = cv2.FONT_HERSHEY_COMPLEX

    def euclaideanDistance(point, point1):
        x, y = point
        x1, y1 = point1
        distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
        return distance
    def get_mouth_region(image, landmarks, mouth_indices, padding):
        h, w, _ = image.shape
        # Get the points corresponding to the mouth region
        mouth_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth_indices]
        # Get the bounding box around the mouth region
        x_min = max(min([p[0] for p in mouth_points]) - padding, 8)  # Add padding and prevent going out of bounds
        y_min = max(min([p[1] for p in mouth_points]) - padding, 8)
        x_max = min(max([p[0] for p in mouth_points]) + padding, w)  # Add padding and prevent going out of bounds
        y_max = min(max([p[1] for p in mouth_points]) + padding, h)
        # Crop the mouth region
        mouth_region = image[y_min:y_max, x_min:x_max]
        return mouth_region, (x_min, y_min, x_max, y_max)
    def get_eye_region(image, landmarks, eye_indices, padding):
        h, w, _ = image.shape
        # Get the points corresponding to the eye region
        eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]

        # Get the bounding box around the eye region
        x_min = max(min([p[0] for p in eye_points]) - padding, 0)  # Add padding and prevent going out of bounds
        y_min = max(min([p[1] for p in eye_points]) - padding, 0)
        x_max = min(max([p[0] for p in eye_points]) + padding, w)  # Add padding and prevent going out of bounds
        y_max = min(max([p[1] for p in eye_points]) + padding, h)

        # Crop the eye region
        eye_region = image[y_min:y_max, x_min:x_max]
        return eye_region, (x_min, y_min, x_max, y_max)

    # Utility functions
    def landmarksDetection(img, results):
        img_height, img_width = img.shape[:2]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                      results.multi_face_landmarks[0].landmark]
        return mesh_coord

    def euclideanDistance(point, point1):
        return math.sqrt((point1[0] - point[0]) ** 2 + (point1[1] - point[1]) ** 2)

    def calculate_angle(line1, line2):
        vec1 = np.array(line1[1]) - np.array(line1[0])
        vec2 = np.array(line2[1]) - np.array(line2[0])
        dot_product = np.dot(vec1, vec2)
        angle_radians = np.arccos(dot_product / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        return np.degrees(angle_radians)

    # Function to classify eye status
    def classify_eye_status(eye_region):
        eye_region_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        eye_region_resized = cv2.resize(eye_region_gray, (224, 224))
        results = model.predict(source=eye_region_resized, imgsz=224, conf=0.25)
        top1_index = results[0].probs.top1
        return top1_index, results[0].names[top1_index]

    # Function to classify mouth status
    def classify_mouth_status(mouth_region):
        mouth_region_gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        mouth_region_resized = cv2.resize(mouth_region_gray, (224, 224))
        results = model2.predict(source=mouth_region_resized, imgsz=224, conf=0.25)
        top1_index = results[0].probs.top1
        return results[0].names[top1_index]

    def front_tilt_ratio(img, landmarks):
        left_most = landmarks[234]
        right_most = landmarks[454]
        up_most = landmarks[94]
        down_most = landmarks[152]
        vertical_distance = euclaideanDistance(up_most, down_most)
        horizontal_distance = euclaideanDistance(left_most, right_most)
        # cv2.line(img, left_most, right_most, utils.GREEN, 2)
        # cv2.line(img, up_most, down_most, utils.GREEN, 2)
        if vertical_distance != 0:
            tilt_ratio = horizontal_distance / vertical_distance
            return tilt_ratio
        else:
            return horizontal_distance

    def caclulate_angle(line1, line2):
        vec1 = np.array(line1[1]) - np.array(line1[0])
        vec2 = np.array(line2[1]) - np.array(line2[0])

        # Calculate the angle using the dot product
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        # Calculate the angle in radians and then convert to degrees
        angle_radians = np.arccos(dot_product / (norm_vec1 * norm_vec2))
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees
    def side_tilt_angle(img, landmarks):
        top_most = landmarks[168]
        down_most = landmarks[4]
        vertical = (top_most[0], top_most[1] + 50)
        line1 = [top_most, down_most]
        line2 = [top_most, vertical]

        my_angle = caclulate_angle(line1, line2)
        # cv2.line(img, top_most, down_most, utils.GREEN, 2)
        # cv2.line(img, top_most, vertical, utils.RED, 2)
        return my_angle

    # Streamlit UI
    st.title("Driver Drowsiness Detection- Uing YOLO11 Classification Models")

    st.write("""This system uses two YOLO 11 models for eye and mouth status classification as 
    Open or Close and Yawning and NOt Yawning respectively.""")

    st.header("Models' Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.image("Images/YOLO11eye.png", caption="Eye Classification")
    with col2:
        st.image("Images/YOLO11yawn.png", caption="Mouth Classification")


    st.write("Click 'Start Detection' to begin or 'Stop Detection' to end.")
    frame_window = st.image([])  # Placeholder for the frame


    if "is_running" not in st.session_state:
        st.session_state["is_running"] = False
    if "frame_count" not in st.session_state:
        st.session_state["frame_count"] = 0



    start_button = st.button("Start Detection")
    stop_button = st.button("Stop Detection")

    if start_button:
        st.session_state["is_running"] = True
    if stop_button:
        st.session_state["is_running"] = False
    if st.session_state["is_running"]:
        cap = cv2.VideoCapture(0)
        while True:
            success, image = cap.read()
            if not success:
                st.error("Camera not working.")
                break
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mesh_coords = landmarksDetection(image, results)

                    # Obtain eye and mouth regions (you can refine this to include your full processing logic)
                    left_eye, left_bbox = get_eye_region(image, face_landmarks.landmark,
                                                         [33, 133, 160, 159, 158, 144, 145, 153, 154, 155], 10)
                    right_eye, right_bbox = get_eye_region(image, face_landmarks.landmark,
                                                           [362, 382, 381, 380, 374, 373, 390, 249, 263, 466], 10)
                    mouth_region, mouth_bbox = get_mouth_region(image, face_landmarks.landmark,
                                                                [61, 291, 81, 178, 185, 40, 39, 37, 0, 17], 10)
                    front_tilt = front_tilt_ratio(image, mesh_coords)
                    side_tilt = side_tilt_angle(image, mesh_coords)
                    # Perform classification
                    mouth_status = classify_mouth_status(mouth_region)
                    left_eye_status, left_label = classify_eye_status(left_eye)
                    right_eye_status, right_label = classify_eye_status(right_eye)
                    left_label=str(left_label)
                    right_label=str(right_label)

                    # Draw bounding boxes and labels
                    cv2.rectangle(image, (mouth_bbox[0], mouth_bbox[1]), (mouth_bbox[2], mouth_bbox[3]), (0, 255, 0), 2)
                    cv2.putText(image, mouth_status, (mouth_bbox[0], mouth_bbox[1] - 10), FONTS, 0.5, (0, 255, 0), 2)

                    # Draw bounding boxes and add text for eye classifications
                    cv2.rectangle(image, (left_bbox[0], left_bbox[1]), (left_bbox[2], left_bbox[3]), (0, 0, 255), 2)
                    cv2.putText(image, left_label.split("_")[0], (left_bbox[0], left_bbox[1] - 10), FONTS, 0.5, (0, 0, 255), 2)

                    cv2.rectangle(image, (right_bbox[0], right_bbox[1]), (right_bbox[2], right_bbox[3]), (0, 0, 255), 2)
                    cv2.putText(image, right_label.split("_")[0], (right_bbox[0], right_bbox[1] - 10), FONTS, 0.5, (0, 0, 255),
                                2)


                    # Set alarm text based on conditions
                    if front_tilt < 1.6 or front_tilt > 3.5 or side_tilt > 20 or (left_eye_status == 0 and right_eye_status == 0) or mouth_status == "yawn":
                        cv2.putText(image, "ALARM!!!", (300, 100), FONTS, 1, (0, 0, 255), 2)
                        # Trigger the alarm sound once when the condition is met
                    if front_tilt < 1.6 or front_tilt > 3.5 or side_tilt > 20:
                        cv2.putText(image, "Driver Distracted", (30, 400), FONTS, 1, (255, 0, 0), 2)

                    utils.colorBackgroundText(image, f'Front tilt : {round(front_tilt, 2)}', FONTS, 0.7, (30, 30), 2,
                                              utils.PINK,
                                              utils.YELLOW)
                    utils.colorBackgroundText(image, f'Side tilt angle : {round(side_tilt, 2)}', FONTS, 0.7, (30, 80),
                                              2,
                                              utils.PINK,
                                              utils.YELLOW)

            frame_window.image(image, channels="BGR")
        cap.release()
    face_mesh.close()
def YOLOV11_Det():
    from ultralytics import YOLO
    import cv2
    st.title("Driver Drowsiness Detection- Uing YOLO11 Detection Model")

    st.write("""This System uses YOLO11 Detection model to detect drowsiness.""")
    st.header("Models' Performance")
    st.image("Images/YOLO11det.png", caption="Drowsiness Detection")
    # 'C:\\Users\\efake\\Downloads\\Detection-epoch 50\\best (9).pt'
    model = YOLO("Models_Final/YOLO11_Det.pt")
    # Streamlit UI
    st.write("Click 'Start Detection' to begin or 'Stop Detection' to end.")
    frame_window = st.image([])
    frame_window = st.image([])  # Placeholder for the frame
    if "is_running" not in st.session_state:
        st.session_state["is_running"] = False
    if "frame_count" not in st.session_state:
        st.session_state["frame_count"] = 0

    start_button = st.button("Start Detection")
    stop_button = st.button("Stop Detection")

    if start_button:
        st.session_state["is_running"] = True
    if stop_button:
        st.session_state["is_running"] = False
    if st.session_state["is_running"]:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from webcam.")
                break

            # Detect drowsiness in the entire frame
            results = model.predict(frame, show=False)

            if results:
                max_confidence = 0  # Track the highest confidence
                best_box = None  # Store the best bounding box
                best_label = None  # Store the label for the highest confidence

                for result in results:
                    for box in result.boxes:
                        confidence = box.conf[0]
                        class_id = int(box.cls[0])
                        label = model.names[class_id]

                        if confidence > max_confidence:
                            max_confidence = confidence
                            best_box = box
                            best_label = label

                # If a box with the highest confidence is found, draw it
                if best_box:
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0])

                    # Draw the bounding box with the highest confidence
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'{best_label} {max_confidence:.2f}', (x1, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            frame_window.image(frame, channels="BGR")
        # Release the webcam and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
def YOLOV10_Det():
    from ultralytics import YOLO
    import cv2
    st.title("Driver Drowsiness Detection- Uing YOLO10 Detection Model")

    st.write("""This System uses YOLO10 Detection model to detect drowsiness.""")
    st.header("Models' Performance")
    st.image("Images/YOLO10det.png", caption="Drowsiness Detection")

    # 'C:\\Users\\efake\\Downloads\\Detection-epoch 50\\best (9).pt'
    model = YOLO("Models_Final/YOLO10_Det.pt")
    # Streamlit UI
    st.title("Driver Drowsiness Detection System")
    st.subheader("Uses YOLO for eye and mouth classification")
    st.write("Click 'Start Detection' to begin or 'Stop Detection' to end.")
    frame_window = st.image([])
    frame_window = st.image([])  # Placeholder for the frame
    if "is_running" not in st.session_state:
        st.session_state["is_running"] = False
    if "frame_count" not in st.session_state:
        st.session_state["frame_count"] = 0

    start_button = st.button("Start Detection")
    stop_button = st.button("Stop Detection")

    if start_button:
        st.session_state["is_running"] = True
    if stop_button:
        st.session_state["is_running"] = False
    if st.session_state["is_running"]:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from webcam.")
                break

            # Detect drowsiness in the entire frame
            results = model.predict(frame, show=False)

            if results:
                max_confidence = 0  # Track the highest confidence
                best_box = None  # Store the best bounding box
                best_label = None  # Store the label for the highest confidence

                for result in results:
                    for box in result.boxes:
                        confidence = box.conf[0]
                        class_id = int(box.cls[0])
                        label = model.names[class_id]

                        if confidence > max_confidence:
                            max_confidence = confidence
                            best_box = box
                            best_label = label

                # If a box with the highest confidence is found, draw it
                if best_box:
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0])

                    # Draw the bounding box with the highest confidence
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'{best_label} {max_confidence:.2f}', (x1, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            frame_window.image(frame, channels="BGR")
        # Release the webcam and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
page_names_to_funcs = {
    "—": intro,
    "Yolo V11 Classification": YOLO11_cls,
    "Yolo V11 Detection": YOLOV11_Det,
    "YOLO V10 Detection": YOLOV10_Det,


}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()