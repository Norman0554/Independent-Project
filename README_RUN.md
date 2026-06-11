# Installation and Launch Guide

This file provides step-by-step instructions on how to set up the dependencies and run the online exam proctoring system on your local machine.

---

## System Requirements
- **OS**: macOS, Windows, or Linux.
- **Python**: Version `3.8` to `3.11` (Recommended: `3.11`).
- **Hardware**: Working webcam and microphone.

---

## Step 1: Install System Dependencies

Depending on your operating system, additional system libraries are required for audio capture (`PyAudio`) and PDF report generation (`pdfkit`).

### For macOS (via Homebrew):
1. Install `portaudio` (required for PyAudio compilation):
   ```bash
   brew install portaudio
   ```
2. Install `wkhtmltopdf` (required for exporting report statistics to PDF format):
   ```bash
   brew install wkhtmltopdf
   ```

### For Windows:
- Download and run the installer for [wkhtmltopdf for Windows](https://wkhtmltopdf.org/downloads.html). Make sure the installation path matches the one specified in `config/config.yaml` under `reporting: wkhtmltopdf_path` (or update it accordingly).

---

## Step 2: Create a Virtual Environment and Install Python Packages

1. Open a terminal and navigate to the project root directory:
   ```bash
   cd /Users/mr.norman/Downloads/exam-cheating-detection-main
   ```
2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   ```
3. Activate the virtual environment:
   - **On macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```
   - **On Windows**:
     ```cmd
     venv\Scripts\activate
     ```
4. Install the required Python packages:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Step 3: Configuration (Optional)

Before running the proctoring client, you can customize detection sensitivity thresholds, recording FPS, log directories, and webcam index in the configuration file:
[config/config.yaml](file:///Users/mr.norman/Downloads/exam-cheating-detection-main/config/config.yaml)

> [!NOTE]
> On macOS, change the `wkhtmltopdf_path` parameter under `reporting:` to your actual installation path, e.g., `/opt/homebrew/bin/wkhtmltopdf` (or leave it blank, and the system will automatically fall back to saving reports in HTML format if the utility is missing).

---

## Step 4: Run the Application

### 1. Start the Proctoring Client:
Run the main script to capture webcam footage, monitor screen activity, record audio, and detect potential violations in real-time:
```bash
python src/main.py
```
*To stop the client, press the **`q`** key inside the webcam window.*
*Upon exit, the recording streams are saved under the `recordings/` folder, and the session report is generated in `reports/generated/`.*

### 2. Start the Analytics Dashboard:
To monitor alerts, violation stats, and live telemetry on a local webpage, open a new terminal tab, activate the virtual environment, and run the Flask server:
```bash
python src/dashboard/app.py
```
Then, open your web browser and navigate to:
```text
http://localhost:5000
```
