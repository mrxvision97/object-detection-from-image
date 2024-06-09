import cv2
import numpy as np
from utils.object_detection import load_model, detect_objects, visualize_detection

def main(image_path):
    model = load_model()
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detections = detect_objects(image_rgb, model)
    image_with_detections = visualize_detection(image, detections)

    cv2.imshow("Detections", image_with_detections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "C:\\Users\\am998\\OneDrive\\Desktop\\End to End\\Image Classifier\\images\\sample_image.jpg"
    main(image_path)



