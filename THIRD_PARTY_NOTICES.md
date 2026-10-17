# Third-party notices

KangEdit's own code is under the license in `LICENSE`. It uses the third-party software below, each under its
own license; the Commons Clause in `LICENSE` does not apply to them.

## Included in this repository

### MobileNetV2 image model

`kangedit/assets/mobilenetv2_embed.onnx`: MobileNetV2 trained on ImageNet, from the
[ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet)
(`mobilenetv2-12.onnx`), cut before its classifier so it outputs the 1280 pooled features. Licensed under the
Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).

### YAMNet sound classification model

`kangedit/assets/yamnet.onnx`: YAMNet (AudioSet sound classes) by Google, from
[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/audio/audio_classifier) (`yamnet.tflite`, from
[TensorFlow Models](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)), converted to
ONNX with tf2onnx. Copyright Google LLC. Licensed under the Apache License, Version 2.0
(https://www.apache.org/licenses/LICENSE-2.0).

### YuNet face detection model

`kangedit/assets/face_detection_yunet_2023mar.onnx`, from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

```
MIT License

Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.```

## Installed by setup or bundled in the packaged builds

| Software | License |
|---|---|
| [Qt for Python (PySide6)](https://www.qt.io/qt-for-python) | LGPL-3.0 |
| [NumPy](https://numpy.org) | BSD-3-Clause |
| [OpenCV (opencv-python)](https://github.com/opencv/opencv-python) | Apache-2.0 (its wheels include FFmpeg under LGPL-2.1) |
| [PyAV](https://pyav.org) | BSD-3-Clause (its wheels include FFmpeg under LGPL-2.1) |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | MIT |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | Unlicense |
| Whisper models (downloaded on first use) | MIT (OpenAI) |

## Downloaded separately

FFmpeg is not part of KangEdit: when it is missing, setup downloads static GPL builds from
[BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) (Windows, Linux) or
[Martin Riedl's build server](https://ffmpeg.martin-riedl.de/) (macOS), covered by FFmpeg's own license.
Official camera LUTs, sounds and music you download are covered by their makers' terms.
