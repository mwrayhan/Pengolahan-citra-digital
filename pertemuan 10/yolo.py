import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

image = cv2.imread("images10.jpeg")
height, width, channels = image.shape

blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.5:
            center_x, center_y, box_width, box_height = detection[0:4] * np.array([width, height, width, height])
            x = int(center_x - box_width / 2)
            y = int(center_y - box_height / 2)
            cv2.rectangle(image, (x, y), (x + int(box_width), y + int(box_height)), (255, 0, 0), 2)
            
cv2.imshow("YOLO Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()