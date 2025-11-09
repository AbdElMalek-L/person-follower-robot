# Fast-and-Accurate-Human-Detection

This project implements two approaches for human detection in images and videos:

1. **HOG (Histogram of Oriented Gradients)** - Traditional computer vision approach using OpenCV's HOG detector
2. **YOLOv8** - Modern deep learning approach using Ultralytics YOLOv8 for more accurate and robust detection



## <u>Some Results</u>

* ![](https://github.com/sovit-123/Fast-and-Accurate-Human-Detection-with-HOG/blob/master/outputs/people1.jpg?raw=true)
* ![](https://github.com/sovit-123/Fast-and-Accurate-Human-Detection-with-HOG/blob/master/outputs/people2.jpg?raw=true)





## <u>The Project Structure</u>

```
├───input
        people1.jpg
        people2.jpg
        people3.jpg
        video1.mp4
        video2.mp4
        video3.mp4
        video4.mp4
├───outputs
│   └───frames
└───src
        hog_detector.py
        hog_detector_vid.py
        hog_detector_webcam.py
        yolo_detector_webcam.py
        yolo_detector_vid.py
```

* After cloning the repository, you need to create the `input` and `outputs` folder.
* You can find all the data in the input folder in the [References](#References) section.



## <u>Executing the Python Files</u>

* `hog_detector.py`: Execute this file from within the `src` folder in the terminal. This detects the people in images inside the `input` folder.
* `hog_detector_vid.py`:  Execution details:
  * `python hog_detector_vid.py --input ../input/video1.mp4 --output ../output/video1_slow.mp4 --speed slow`: Use this execution command to run slow but accurate video detection algorithm.
  * `python hog_detector_vid.py --input ../input/video1.mp4 --output ../output/video1_fast.mp4 --speed fast`: Use this command to execute a little bit less accurate but fast video detection algorithm.

* `hog_detector_webcam.py`: Run realtime detection from your webcam using HOG:
  * `python src/hog_detector_webcam.py` (default device 0, width 640, fast mode)
  * Options: `-d/--device` to choose camera index, `-w/--width` to set frame width, `-s/--speed` choose `fast` or `slow`.

### YOLOv8 Detection

YOLOv8 provides more accurate and robust people detection using deep learning. To use YOLOv8 detection:

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run YOLOv8 webcam detection:
   ```
   python src/yolo_detector_webcam.py
   ```
   Press 'q' to quit.

3. Run YOLOv8 video detection:
   ```
   python src/yolo_detector_vid.py input/video1.mp4 --output outputs/video1_yolo.mp4
   ```
   The `--output` parameter is optional. If not provided, the video will only be displayed.

## Setup (create virtual environment)

It's recommended to run the project inside a virtual environment. From PowerShell in the repository root you can create and populate `.venv` by running:

```powershell
.\create_venv.ps1
```

This will create `.venv` and install packages from `requirements.txt`. After that activate the venv in PowerShell with:

```powershell
& .\.venv\Scripts\Activate.ps1
```

Then run the webcam script:

```powershell
python .\src\hog_detector_webcam.py
```



## <u>References</u>

* **Credits and Citations**:
  * `input/people1.jpg`: Image by <a href="https://pixabay.com/photos/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=438393">Free-Photos</a> from <a href="https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=438393">Pixabay</a>.
    * Link: https://pixabay.com/photos/urban-people-crowd-citizens-438393/.
  * `input/people2.jpg`: http://www.cbc.ca/natureofthings/content/images/episodes/pompeiipeople_listical.jpg.
  * `input/people3.jpg`: https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.us0yaLcftx1jwMQ-tcw34gHaEU%26pid%3DApi&f=1.
  * `input/video1.mp4`: https://pixabay.com/videos/people-commerce-shop-busy-mall-6387/.
  * `input/video2.mp4`: https://pixabay.com/videos/pedestrians-road-city-cars-traffic-1023/.
  * `input/video3.mp4`: Video by <a href="https://pixabay.com/users/surdumihail-3593622/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=6096">Mihai Surdu</a> from <a href="https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=6096">Pixabay</a>.
    * Link: https://pixabay.com/videos/park-old-people-people-old-senior-6096/.
  * `input/video4.mp4`:
    * Link: https://www.pexels.com/video/athletes-warming-up-1585619/.
  * `input/video5.mp5`: 
    * Link: https://www.youtube.com/watch?v=NyLF8nHIquM.
