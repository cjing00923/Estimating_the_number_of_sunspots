Sunspot Segmentation and Analysis Pipeline
This project provides an automated pipeline for sunspot image analysis, including segmentation, umbra and penumbra separation, sunspot group detection, and butterfly diagram generation.

Project Structure and Functionality
1. Sunspot Segmentation Model Training and Inference
UNet_slice_train.py
Trains a U-Net model for semantic segmentation of sunspots from solar disk images.

predict_all.py
Applies the trained U-Net model to segment sunspots in input images.

2. Umbra and Penumbra Separation (Umbra_and_Penumbra folder)
peak_all.py
Detects intensity peaks in each segmented sunspot region to determine the presence of a penumbra.

k_all.py
Applies KMeans clustering to sunspots with penumbrae to separate the umbra and penumbra regions, enabling single sunspot counting.

3. Sunspot Group Detection
Sunspot group detection is performed using a YOLOv5 model. The detection code is available at:
[cjing00923-sunspot_group_detection_yolov5](https://github.com/cjing00923/cjing00923-sunspot_group_detection_yolov5)

4. Butterfly Diagram Generation (Butterfly folder)
find_box.py
Extracts the bounding box coordinates of each sunspot group detected by YOLO.

box_erzhi.py
Converts the detected bounding boxes into binary mask images.

mask.py
Overlays the YOLO-detected binary masks onto the U-Net segmented sunspot results to obtain detailed contours of each sunspot group.

area.py
Calculates the area of each sunspot group.

latitude_box.py
Computes the latitude of each sunspot group.

merge.py
Merges area and heliographic latitude/longitude information into a single output file.

draw.py
Draws the sunspot butterfly diagram based on the area and latitude/longitude data of each sunspot group.

Usage Guide
Train the sunspot segmentation model using UNet_slice_train.py.

Segment sunspots on solar images using predict_all.py.

In the Umbra_and_Penumbra/ folder, run peak_all.py and k_all.py to separate umbra and penumbra.

Detect sunspot groups using the YOLO model from the sunspot group detection repository.

In the Butterfly/ folder, run the following scripts sequentially to analyze sunspot groups and generate the butterfly diagram:

find_box.py

box_erzhi.py

mask.py

area.py

latitude_box.py

merge.py

draw.py
