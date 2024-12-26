# IMageEval : Image Quality Evaluation and Enhancement Tool

This document combines the evaluation metrics, enhancement logic, and setup instructions to guide users in using the Image Quality Evaluation and Enhancement Tool effectively.

---

## 1. Overview

The Image Quality Evaluation and Enhancement Tool is designed to assess and improve image quality systematically. It evaluates images based on resolution, clarity, and background quality, and enhances them using advanced sharpening, upscaling, and background replacement techniques.

---

## 2. Evaluation Metrics

The tool uses three primary metrics: **Resolution**, **Clarity**, and **Background Quality**.

### 2.1 Resolution Check
- **Description**: Ensures that the image meets the minimum resolution requirements.
- **Threshold**: Default minimum resolution is 1024x1024 pixels.
- **Implementation**:
  ```python
  def check_resolution(image, min_resolution=(1024, 1024)):
      if len(image.shape) == 2:  # Grayscale image
          height, width = image.shape
      else:  # Color image
          height, width, _ = image.shape

      if height < min_resolution[1] or width < min_resolution[0]:
          return False, "Low resolution"
      return True, "Resolution OK"
  ```

### 2.2 Clarity Check
- **Description**: Evaluates the sharpness of the image using the variance of the Laplacian.
- **Threshold**: Default sharpness threshold is 100.0.
- **Implementation**:
  ```python
  def check_clarity(image, sharpness_threshold=100.0):
      gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
      if laplacian_var < sharpness_threshold:
          return False, "Blurry", laplacian_var
      return True, "Clarity OK", laplacian_var
  ```

### 2.3 Background Check
- **Description**: Determines whether the image has a plain background (e.g., white).
- **Threshold**: Default maximum unique colors for a simple background is 5.
- **Implementation**:
  ```python
  def check_background(image_path, tolerance=10, max_unique_colors=5):
      img = Image.open(image_path)
      bg_removed = remove(img)
      bg_array = np.array(bg_removed.resize((100, 100)))
      unique_colors = len(np.unique(bg_array.reshape(-1, bg_array.shape[2]), axis=0))
      if unique_colors > max_unique_colors:
          return False, "Complex background"
      return True, "Background OK"
  ```

### Evaluation Workflow
The evaluation function integrates the above checks:
```python
def evaluate_image(image_path, min_resolution=(1024, 1024), sharpness_threshold=100.0):
    image = cv2.imread(image_path)
    issues = []
    metadata = {}

    # Resolution Check
    resolution_ok, resolution_message = check_resolution(image, min_resolution)
    if not resolution_ok:
        issues.append(resolution_message)
    metadata["resolution"] = image.shape[:2]

    # Clarity Check
    clarity_ok, clarity_message, sharpness = check_clarity(image, sharpness_threshold)
    if not clarity_ok:
        issues.append(clarity_message)
    metadata["sharpness_score"] = sharpness

    # Background Check
    background_ok, background_message = check_background(image_path)
    if not background_ok:
        issues.append(background_message)

    status = "Good" if not issues else "Needs Improvement"
    return {
        "status": status,
        "issues": issues,
        "metadata": metadata,
    }
```

---

## 3. Enhancement Logic

The tool enhances image quality through sharpening, upscaling, and background replacement.

### 3.1 Sharpening
- **Description**: Enhances image details using a sharpening kernel.
- **Implementation**:
  ```python
  def sharpen_image(image):
      kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
      return cv2.filter2D(image, -1, kernel)
  ```

### 3.2 Upscaling
- **Description**: Improves resolution using the ESRGAN model.
- **Implementation**:
  ```python
  def upscale_image_with_esrgan(image, model):
      image_tensor = TF.to_tensor(image).unsqueeze(0)
      with torch.no_grad():
          enhanced_tensor = model(image_tensor)
      enhanced_image = TF.to_pil_image(enhanced_tensor.squeeze(0))
      return enhanced_image
  ```

### 3.3 Background Replacement
- **Description**: Replaces the background with a plain white background.
- **Implementation**:
  ```python
  def replace_background(image):
      bg_removed = remove(image)
      if bg_removed.mode != "RGBA":
          bg_removed = bg_removed.convert("RGBA")
      white_bg = Image.new("RGBA", bg_removed.size, (255, 255, 255, 255))
      white_bg.paste(bg_removed, (0, 0), bg_removed)
      return white_bg
  ```

### Enhancement Workflow
The complete enhancement logic:
```python
def enhance_image(image_path, save_path, esrgan_model):
    image = cv2.imread(image_path)
    sharpened = sharpen_image(image)
    pil_image = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    upscaled = upscale_image_with_esrgan(pil_image, esrgan_model)
    enhanced_image = replace_background(upscaled)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    enhanced_image.save(save_path)
    return save_path
```

---

## 4. Steps to Run the Tool

### 4.1 Set Up the Environment

1. **Install Python and Dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # For Linux/Mac
   pip install -r requirements.txt
   ```

2. **Configure the PYTHONPATH**
   ```bash
   export PYTHONPATH=$(pwd)   # For Linux/Mac
   set PYTHONPATH=%cd%       # For Windows
   ```

3. **Database Configuration**
   Update `DATABASE_CONFIG` in `app/config.py` with PostgreSQL credentials:
   ```python
   DATABASE_CONFIG = {
       "host": "localhost",
       "port": 5432,
       "user": "your_username",
       "password": "your_password",
       "database": "your_database_name"
   }
   ```
   Initialize tables:
   ```bash
   python scripts/database_setup.py
   ```

### 4.2 Fetch and Process Images

1. **Fetch Images from Database**
   ```bash
   PYTHONPATH=. python scripts/fetch_images.py
   ```

2. **Process Images**
   ```bash
   PYTHONPATH=. python scripts/batch_process.py
   ```

### 4.3 Testing and Validation

- Run unit tests:
  ```bash
  PYTHONPATH=$(pwd) pytest -v tests/test_pipeline.py
  ```
- Inspect enhanced images in `data/enhanced/`.

### 4.4 Manage Outputs

- Enhanced images are saved locally and updated in the database with quality metrics and enhancement details.
- Use SQL queries to validate database entries:
  ```sql
  SELECT * FROM enhanced_images;
  ```

---

## 5. Troubleshooting

- **Database Errors**: Verify credentials and PostgreSQL status.
- **Missing Dependencies**: Run `pip install -r requirements.txt`.
- **Performance Issues**: Use parallel processing for large datasets.

Logs are available for monitoring execution and debugging.

---

By following this guide, users can effectively evaluate, enhance, and manage image quality using this tool.

