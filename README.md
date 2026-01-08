 SlicerFootSegmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![3D Slicer](https://img.shields.io/badge/3D%20Slicer-Extension-blue)](https://www.slicer.org/)

AI-powered automatic 3D foot segmentation extension for 3D Slicer.

![Extension Preview](Screenshots/preview.png)

 Description

This extension provides automatic segmentation of foot structures from 3D medical images (NIfTI/DICOM) using a deep learning model. The model is based on 3D U-Net architecture and converted to ONNX format for optimal performance.

 Features

- 🔬 Automatic Segmentation: One-click foot segmentation
- ⚡ Fast Inference: ONNX Runtime for optimized performance
- 🖥️ GPU Support: Optional CUDA acceleration
- 📊 Adjustable Parameters: Threshold and overlap settings
- 🌍 Multi-language UI: Turkish interface for medical professionals

 Installation

 Method 1: Extension Manager (Recommended)

1. Open 3D Slicer
2. Go to View → Extension Manager
3. Search for "Foot Segmentation"
4. Click Install
5. Restart 3D Slicer

 Method 2: Manual Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/cuneytozdemir/FootSegmentation.git
   ```

2. In 3D Slicer, go to Edit → Application Settings → Modules

3. Add the path to the `FootSegmentation` folder

4. Restart 3D Slicer

 Usage

1. Load a volume (File → Add Data)
2. Open Modules → Segmentation → Foot Segmentation
3. Select your input volume
4. Click Start Segmentation
5. The segmentation result will appear automatically

 Requirements

- 3D Slicer 5.0 or later
- onnxruntime (automatically installed)

 Model Information

| Property | Value |
|----------|-------|
| Architecture | 3D U-Net |
| Input Size | 64 × 128 × 128 |
| Format | ONNX |
| Training Data | Foot CT scans |

 Screenshots

| Input | Output |
|-------|--------|
| ![Input](Screenshots/input.png) | ![Output](Screenshots/output.png) |

 Citation

If you use this extension in your research, please cite:

```bibtex
@software{foot_segmentation_slicer,
  author = {Cuneyt OZDEMİR, Mehmet Ali GEDİK},
  title = {SlicerFootSegmentation: AI-powered 3D Foot Segmentation},
  year = {2026},
  url = {https://github.com/cuneytozdemir/FootSegmentation}
}
```

 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

 Acknowledgments

- [3D Slicer](https://www.slicer.org/) community
- ONNX Runtime team
- TensorFlow team

 Contact

- Author: Cüneyt ÖZDEMİR
- Email: cuneytozdemir33@gmail.com
- Institution: Siirt Üniversitesi

