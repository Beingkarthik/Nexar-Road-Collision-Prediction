# 🚗 RoadLens — AI-Based Road Collision & Near-Miss Prediction Using Computer Vision

<<<<<<< HEAD
**An interactive deep-learning application for analyzing dashcam videos, identifying potentially concerning frames, and visualizing model-generated collision-risk scores.**
=======
[🚀 Launch Live App](https://roadlens-collision-prediction.streamlit.app/)


## Overview
This application analyzes road videos using a trained ResNet18 model.
>>>>>>> 62c91c4 (docs: add live application link)

RoadLens explores how computer vision can support road-safety research by processing video frames with a trained ResNet18-based classifier and presenting the results through an interactive Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Interface-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![GitHub](https://img.shields.io/badge/Source-GitHub-181717?logo=github&logoColor=white)](https://github.com/Beingkarthik/Nexar-Road-Collision-Prediction)

> **Project status:** Local application development. Deployment and public demo links will be added after hosting is configured.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Technology Stack](#-technology-stack)
- [Model Information](#-model-information)
- [Application Preview](#-application-preview)
- [Installation and Setup](#️-installation-and-setup)
- [Using the Application](#-using-the-application)
- [Results and Evaluation](#-results-and-evaluation)
- [Project Structure](#-project-structure)
- [Limitations and Responsible Use](#️-limitations-and-responsible-use)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)
- [Author](#-author)

## 🔍 Project Overview

Road collisions and near-miss situations are important subjects in intelligent transportation and road-safety research. Dashcam footage provides a way to study road scenes, but reviewing long videos frame by frame can be time-consuming.

**RoadLens** is a computer-vision prototype that explores automated analysis of dashcam video using deep learning.

The application samples frames from a selected video, processes them with a trained ResNet18-based binary classifier, and displays the resulting scores and threshold-based flags in an interactive dashboard.

Users can inspect the analysis summary, explore score changes across sampled frames, review annotated video output, and export supported analysis results.

### What makes this project useful?

- Converts video analysis into a visual, interactive workflow.
- Makes frame-level model outputs easier to inspect.
- Provides configurable sampling and threshold settings.
- Combines deep-learning inference with video processing and reporting.
- Demonstrates how a computer-vision model can be integrated into a practical Python application.

## 🎯 Problem Statement

Potentially dangerous road situations may develop over short periods, and manually reviewing lengthy dashcam footage requires time and attention.

This project investigates whether a trained deep-learning classifier can help identify frames that warrant further inspection.

The objective is to build a reproducible video-analysis workflow that samples dashcam footage, generates model scores, highlights selected frames, and presents the results in a usable interface.

## 🎯 Project Objectives

1. Process user-provided dashcam videos through a computer-vision pipeline.
2. Apply a trained ResNet18-based model to sampled video frames.
3. Generate and visualize frame-level class scores.
4. Flag frames according to a configurable threshold.
5. Present analysis summaries and score timelines in a Streamlit dashboard.
6. Generate annotated video output and downloadable reports where supported.
7. Provide a foundation for future evaluation, temporal modeling, and road-safety research.

## ✨ Key Features

### 🎥 Video Input and Processing

- Upload a supported video file for analysis.
- Select an available sample video.
- Sample frames at a configurable interval.
- Set a maximum number of frames to process.
- Generate annotated video output when processing completes.

### 🧠 Deep-Learning Inference

- Use the existing trained ResNet18-based checkpoint.
- Produce model scores for sampled frames.
- Apply a configurable threshold to flag selected frames.
- Inspect outputs without retraining the model during ordinary inference.

### 📊 Interactive Analysis Dashboard

- Review the number of frames analyzed.
- Examine the number and percentage of frames flagged.
- View the peak positive-class score and other supported summary statistics.
- Explore score changes through a timeline visualization.
- Compare results under different analysis settings.

### 📁 Reporting and Export

- Download a text-based analysis summary.
- Export frame-level results to CSV where supported.
- Review the annotated video output.
- Use the generated results for further inspection and experimentation.

> Feature availability depends on the selected input, processing outcome, and current application implementation.

## ⚙️ How It Works

The application follows a video-inference pipeline.

```text
              Dashcam Video
                    |
                    v
          Video Input and Validation
                    |
                    v
             Frame Sampling
                    |
                    v
       Frame Preprocessing and Resizing
                    |
                    v
       Trained ResNet18-Based Classifier
                    |
                    v
          Frame-Level Model Scores
                    |
                    v
       Configurable Threshold Evaluation
                    |
                    v
       Analysis Summary and Visualizations
                    |
          +---------+---------+
          |         |         |
          v         v         v
      Timeline   Annotated   CSV / Text
       Chart       Video      Reports
```

### Pipeline stages

**1. Video input**

The user selects a sample video or uploads a supported dashcam video.

**2. Frame sampling**

The application extracts selected frames based on the configured sampling settings.

**3. Preprocessing**

Frames are prepared in the format expected by the trained model.

**4. Model inference**

The ResNet18-based classifier generates class scores for the processed frames.

**5. Threshold-based flagging**

The application compares the relevant score against the selected threshold to determine which frames are flagged.

**6. Result presentation**

The dashboard displays summary statistics, visualizations, and supported downloadable outputs.

## 🧠 Model Information

| Property | Details |
|---|---|
| Model architecture | ResNet18-based classifier |
| Deep-learning framework | PyTorch |
| Task | Binary frame classification |
| Input | Sampled video frames |
| Output | Frame-level class scores |
| Decision mechanism | Configurable score threshold |
| Checkpoint | `models/improved_resnet18.pth` |
| Inference interface | Streamlit application |

### Understanding the model output

The model produces a score associated with its positive class. The application uses this output for visualization and threshold-based frame flagging.

A flagged frame is a frame selected by the model and configured threshold for further inspection. It is not proof that a collision will occur.

**Important:** The model's output must not be interpreted as a calibrated probability of a future collision. Its reliability depends on the training data, evaluation methodology, input conditions, and model behavior.

The application is an inference interface for an existing checkpoint; uploading a video does not retrain the model.

## 🖥️ Application Preview

Application screenshots will be added after the deployment and final UI review.

### 1. Dashboard Overview

*Planned screenshot: dashboard metrics and video selection.*

`[Dashboard screenshot to be added]`

### 2. Video Analysis Results

*Planned screenshot: analyzed video and frame-level flags.*

`[Video analysis screenshot to be added]`

### 3. Score Timeline

*Planned screenshot: model scores across sampled frames.*

`[Score timeline screenshot to be added]`

### 4. Analysis Report

*Planned screenshot: summary statistics and downloadable outputs.*

`[Analysis report screenshot to be added]`

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and inference workflow |
| PyTorch | Deep-learning inference |
| ResNet18 | Frame-level image classification architecture |
| OpenCV | Video decoding, frame extraction, and video processing |
| Streamlit | Interactive web application |
| NumPy | Numerical processing where used |
| Pandas | Tabular analysis and CSV handling where used |
| Matplotlib / charting library | Visualization, according to the implementation |
| Git and GitHub | Version control and source hosting |
| Visual Studio Code | Local development environment |

The exact installed package versions are defined by the project's `requirements.txt` file.

## 🚀 Installation and Setup

### Prerequisites

- Windows, Linux, or macOS
- Python version compatible with the project's dependencies
- Git
- Internet access to install dependencies
- The trained model checkpoint
- Sufficient memory and storage to process the selected video

### 1. Clone the repository

```bash
git clone https://github.com/Beingkarthik/Nexar-Road-Collision-Prediction.git
cd Nexar-Road-Collision-Prediction
```

### 2. Create a virtual environment

**Windows — Command Prompt:**

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install the dependencies

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation fails, select a compatible installation for your Python version and hardware using the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Verify the model checkpoint

Ensure the following file exists:

```text
models/improved_resnet18.pth
```

The application expects the checkpoint at the configured model path. If it is not distributed with a future release, obtain it from an authorized source.

### 5. Launch the application

```cmd
streamlit run app.py
```

Open the local URL displayed in the terminal. The default address is typically:

**http://localhost:8501**

Keep the terminal running while using the application.

## 🎮 Using the Application

1. Launch the Streamlit application.
2. Select a provided sample video or upload a supported video.
3. Configure the frame-sampling interval and maximum frame count.
4. Choose the score threshold for flagging frames.
5. Run the video analysis.
6. Review the dashboard's summary metrics.
7. Explore the score timeline and annotated video.
8. Download available CSV or text reports for further analysis.

### Configuration concepts

| Setting | Purpose |
|---|---|
| Score threshold | Controls which frame scores are flagged |
| Frame-sampling interval | Controls how frequently frames are analyzed |
| Maximum frames | Limits the number of frames processed |

Exact setting names and permitted values depend on the current application interface.

## 📈 Results and Evaluation

The dashboard presents descriptive statistics about the video analysis, including frame counts and model scores. These outputs help users understand what the model produced for a particular input.

### Available analysis outputs

| Output | Description |
|---|---|
| Frames analyzed | Number of sampled frames processed |
| Frames flagged | Number of frames meeting the selected threshold |
| Flagged-frame rate | Proportion of analyzed frames flagged |
| Peak positive-class score | Highest positive-class score among processed frames |
| Score timeline | Visual representation of scores across sampled frames |
| Annotated video | Video output with application-generated annotations |
| CSV report | Exportable analysis data where supported |
| Text summary | Downloadable summary where supported |

### Quantitative model evaluation

A professional model evaluation should report results on a clearly defined held-out dataset and document the evaluation procedure.

Useful metrics for a binary classifier include:

- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC or PR-AUC, where appropriate

For a collision-prediction task, additional evaluation should investigate the lead time before an event, false alarms, missed events, and performance across different road and environmental conditions.

**Verified benchmark metrics have not been included in this README because no confirmed evaluation report is currently documented here.** Dashboard summaries alone are not sufficient to establish predictive accuracy.

## 📂 Project Structure

```text
Nexar-Road-Collision-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── models/
│   └── improved_resnet18.pth
│
├── demo/
│   ├── sample_road_video.mp4
│   └── videos/
│       └── manifest.csv
│
├── src/
│   ├── components/
│   ├── services/
│   └── utils/
│
└── tests/
```

This structure represents the core files currently prepared for the repository. Some folders may contain additional local files.

The large experimental notebook and selected development backups are excluded from the initial GitHub upload. The local virtual environment, generated outputs, and excluded demo videos are not part of this tracked structure.

## ⚠️ Limitations and Responsible Use

RoadLens is a research and demonstration prototype, not a validated collision-avoidance system.

- **No guaranteed prediction:** A model flag does not guarantee that a collision will happen.
- **No calibrated probability:** Model scores should not be treated as real-world collision probabilities.
- **Temporal limitations:** Sparse frame sampling can miss brief events and changes between analyzed frames.
- **Environmental variation:** Lighting, weather, camera angle, motion blur, and occlusion may affect outputs.
- **Model generalization:** Performance on unseen locations, cameras, and road conditions has not been established by the dashboard alone.
- **False positives and negatives:** Concerning frames may be missed, and harmless frames may be flagged.
- **Threshold sensitivity:** Changing the threshold affects the number of flagged frames.
- **Processing constraints:** Video resolution, length, frame-sampling settings, and available hardware influence processing time.

Do not use this prototype as a substitute for driver attention, certified vehicle-safety technology, or professional safety assessments. It should not be used to make real-time driving decisions.

## 🔮 Future Improvements

Potential next steps for the project include:

- Evaluate the classifier using a documented held-out test set.
- Report precision, recall, F1-score, and confusion-matrix results.
- Investigate temporal models that use information across consecutive frames.
- Study event-level detection and collision lead-time evaluation.
- Measure false-alarm rates and missed-event rates.
- Test robustness across lighting, weather, road types, and camera viewpoints.
- Improve inference speed and resource usage.
- Expand automated testing and reproducible experiment configuration.
- Deploy the application and publish a public demo.
- Add application screenshots and a demonstration video.
- Improve documentation for model provenance, data permissions, and reproducibility.

These items are proposed improvements, not claims of currently implemented functionality.

## 🤝 Contributing

Suggestions, bug reports, and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your change.
4. Commit your changes with a descriptive message.
5. Open a pull request explaining the change.

Please avoid committing credentials, private datasets, unlicensed video files, or large generated artifacts.

## 🙏 Acknowledgements

This project uses open-source tools and deep-learning technologies, including:

- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)
- [GitHub](https://github.com/)

The project is related to the research area of dashcam-based collision prediction. Appropriate attribution and license compliance should be maintained for any datasets, pretrained weights, checkpoints, and third-party assets used.

## 📄 License

A license for this repository has not yet been specified in this README. Before publishing or redistributing the project, choose and add an appropriate `LICENSE` file.

Dataset videos, trained weights, and other third-party materials may have separate terms. Confirm those terms before redistribution.

## 👨‍💻 Author

**Karthik**

GitHub: [@Beingkarthik](https://github.com/Beingkarthik)

Project repository: [Nexar Road Collision Prediction](https://github.com/Beingkarthik/Nexar-Road-Collision-Prediction)

---

**RoadLens — Exploring safer roads through computer vision.**
