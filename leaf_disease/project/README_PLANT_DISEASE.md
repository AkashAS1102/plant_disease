# Plant Disease Recognition System

A complete end-to-end AI-powered plant disease recognition web application featuring:
- **Machine Learning Model** trained on the PlantVillage dataset
- **FastAPI Backend** with comprehensive plant care knowledge base
- **Beautiful Modern Frontend** with responsive design

## Features

- Automatic PlantVillage dataset downloading and model training
- CNN-based image classification with data augmentation
- RESTful API with CORS support
- Mock prediction fallback for immediate UI testing
- Comprehensive treatment recommendations including:
  - Plant disease details
  - Fertilizer recommendations
  - Treatment and medicine guidance
- Responsive, production-ready frontend design
- Modern UI with smooth animations and transitions

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Modern web browser (Chrome, Firefox, Safari, or Edge)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow (Deep Learning framework)
- TensorFlow Datasets (for automatic dataset downloading)
- FastAPI (Web framework)
- Uvicorn (ASGI server)
- Pillow (Image processing)
- NumPy (Numerical computing)

## Usage Guide

### Step 1: Train the Model (Optional but Recommended)

Train the CNN model on the PlantVillage dataset:

```bash
python train_model.py
```

**What happens:**
- Automatically downloads the PlantVillage dataset (~870MB)
- Preprocesses and augments the training data
- Trains a CNN model for 3 epochs (quick execution)
- Saves the model as `plant_disease_model.h5`

**Training Time:** 10-30 minutes depending on your hardware (faster with GPU)

**Note:** The API works without training by using mock predictions, perfect for testing the UI immediately.

### Step 2: Start the Backend API

Launch the FastAPI server:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload
```

**The API will start on:** `http://localhost:8000`

**Endpoints:**
- `GET /` - Health check and API status
- `POST /predict` - Upload image for disease prediction
- `GET /docs` - Interactive API documentation (Swagger UI)

### Step 3: Open the Frontend

Open `plant_disease_app.html` in your web browser:

```bash
# On macOS
open plant_disease_app.html

# On Linux
xdg-open plant_disease_app.html

# On Windows
start plant_disease_app.html
```

Or simply double-click the `plant_disease_app.html` file.

## How to Use the Application

1. **Click "Scan Leaf"** button
2. **Select an image** of a plant leaf from your device
3. **Wait for analysis** (2-3 seconds)
4. **View results** with:
   - Disease identification
   - Confidence percentage
   - Plant details
   - Fertilizer recommendations
   - Treatment and medicine guidance

## Project Structure

```
.
├── train_model.py              # ML model training script
├── main.py                     # FastAPI backend server
├── plant_disease_app.html      # Frontend web application
├── requirements.txt            # Python dependencies
├── plant_disease_model.h5      # Trained model (created after training)
└── README_PLANT_DISEASE.md     # This file
```

## Supported Plant Diseases

The model recognizes 10 common plant diseases:

1. **Tomato Early Blight** - Fungal disease with concentric ring patterns
2. **Tomato Late Blight** - Devastating water mold infection
3. **Potato Late Blight** - Same pathogen as Irish Potato Famine
4. **Potato Early Blight** - Brown spots with target-like patterns
5. **Apple Scab** - Fungal spots on leaves and fruit
6. **Grape Black Rot** - Mummified berry disease
7. **Corn Common Rust** - Brown pustules on leaves
8. **Pepper Bacterial Spot** - Dark spots with yellow halos
9. **Strawberry Leaf Scorch** - Purple spots with tan centers
10. **Healthy** - No disease detected

## API Response Format

```json
{
  "predicted_class": "Tomato___Early_Blight",
  "confidence": 87.45,
  "plant_details": "Early blight is a common fungal disease...",
  "fertilizer": "Use a balanced NPK fertilizer with ratio 10-10-10...",
  "medicine": "Apply copper-based fungicides such as Bordeaux mixture..."
}
```

## Design Highlights

### Frontend Features
- **Massive Typography:** 6xl to 9xl font sizes for bold headlines
- **Inline Leaf Image:** Pill-shaped green leaf integrated into title
- **Neon Green CTA:** Eye-catching primary button with green glow effect
- **Warm Gradient Background:** Subtle beige gradient (fdfbf7 to f4eee1)
- **Three-Card Layout:** Color-coded cards for Details, Fertilizer, Medicine
- **Smooth Animations:** Fade-in effects and hover states
- **Fully Responsive:** Mobile-first design with breakpoints

### Backend Features
- **Automatic Fallback:** Works without trained model for instant testing
- **Comprehensive Database:** 10 plant diseases with detailed information
- **CORS Enabled:** Allows frontend requests from any origin
- **Error Handling:** Graceful degradation with informative messages
- **RESTful Design:** Clean API structure with proper HTTP methods

## Troubleshooting

### API Connection Error

**Problem:** Frontend shows "Error analyzing image"

**Solution:**
- Ensure the backend is running: `python main.py`
- Check that the API is accessible at `http://localhost:8000`
- Open browser console (F12) for detailed error messages

### Model Not Loading

**Problem:** API console shows "Model file not found"

**Solution:**
- This is normal if you haven't trained the model yet
- The API will use mock predictions automatically
- To use the real model, run: `python train_model.py`

### TensorFlow Installation Issues

**Problem:** pip install tensorflow fails

**Solution:**
```bash
# Try with specific version
pip install tensorflow==2.14.0

# Or use CPU-only version
pip install tensorflow-cpu
```

### Dataset Download Fails

**Problem:** PlantVillage dataset won't download

**Solution:**
- Check your internet connection
- Ensure you have ~1GB free disk space
- The dataset will be cached in `~/tensorflow_datasets/`

## Performance Tips

1. **Use GPU:** TensorFlow automatically uses GPU if available (CUDA-enabled)
2. **Increase Training Epochs:** Edit `train_model.py` and change `epochs=3` to `epochs=10` for better accuracy
3. **Batch Size:** Reduce if you get memory errors, increase for faster training
4. **Image Quality:** Upload clear, well-lit photos for best results

## Technology Stack

### Machine Learning
- TensorFlow 2.x - Deep learning framework
- TensorFlow Datasets - Dataset management
- CNN Architecture - Convolutional Neural Network

### Backend
- FastAPI - Modern Python web framework
- Uvicorn - Lightning-fast ASGI server
- Pillow - Image processing library

### Frontend
- HTML5 - Semantic markup
- Tailwind CSS - Utility-first styling
- Vanilla JavaScript - No frameworks needed
- Inter Font - Premium Google Font

## License

This project is for educational and demonstration purposes.

## Disclaimer

This tool is designed for educational purposes and should not replace professional agricultural advice. For critical crop decisions, always consult with agricultural experts, plant pathologists, or local extension services.

## Contributing

Feel free to fork, modify, and enhance this project. Contributions are welcome!

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the API documentation at `http://localhost:8000/docs`
3. Inspect browser console for frontend errors (F12)

---

**Built with TensorFlow, FastAPI, and Modern Web Technologies**
