# How to Run the Face Mask Detection App

This project uses a unified architecture where **Flask serves both the backend ML logic and the frontend UI** at the same time. You do not need to run a separate frontend server (like Node.js or React). 

Here are the step-by-step instructions to get the application running locally.

## 1. Prerequisites

Make sure you have Python installed. The project is currently configured to use a virtual environment located in the `venv_new` (or `venv`) directory.

## 2. Activate the Virtual Environment

Before running the application or installing dependencies, activate the virtual environment:

**On macOS / Linux:**
```bash
source venv_new/bin/activate
```

**On Windows:**
```cmd
.\venv_new\Scripts\activate
```

## 3. Install Dependencies

Once the virtual environment is active, install all required packages:

```bash
pip install -r requirements.txt
```

> **Note for Python 3.14+ users:** If you face issues installing `tensorflow` natively via the requirements file, you may need to install the nightly build of TensorFlow to ensure compatibility:
> ```bash
> pip install tf-nightly tf-keras
> ```

## 4. Run the Application

Start the Flask server, which will initialize the deep learning model and start serving the web pages:

```bash
python app.py
```

You should see output indicating the model is loading, followed by:
```text
  Face Mask Detection Server
  Open  http://localhost:5001  in your browser.
```

## 5. View the App (Frontend)

1. Open your web browser (Chrome, Edge, Safari, etc.).
2. Navigate to **http://localhost:5001**.
3. Click the **"Start Detection"** button on the UI.
4. Your browser may prompt you for **Camera Permissions**—make sure to allow it so the OpenCV backend can access your webcam feed.

## Troubleshooting
- **Address already in use:** If port `5001` is taken, you can change the port number at the very bottom of `app.py` in the `app.run(..., port=5001)` line.
- **Camera not turning on:** Ensure no other application (like Zoom or Teams) is actively using your camera, as OpenCV requires exclusive access to the webcam hardware.
