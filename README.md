# Unified YOLO for Smart Manufacturing Safety

A unified deep learning solution for real-time safety monitoring in manufacturing environments, combining PPE (Personal Protective Equipment) detection and fire safety monitoring using a dual-head YOLO architecture.

## 🎯 Features

- **Unified Detection System**: Single model architecture for detecting both PPE compliance and fire hazards
- **Real-time Processing**: Optimized for real-time video processing in industrial environments
- **Dual-head Architecture**: Specialized detection heads for PPE and fire safety
- **Industrial Focus**: Designed specifically for manufacturing and industrial settings
- **High Accuracy**: Enhanced detection accuracy through specialized training

## 🛠 Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA compatible GPU (recommended)
- Additional dependencies in `requirements.txt`

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/azimjaan21/unified_safety.git
cd unified_safety
```

2. Create and activate a virtual environment:
```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux/Mac
source env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

1. Configure your settings in `configs/`:
   - `dual_data.yaml`: Dataset configuration
   - `yolo11m_dualhead.yaml`: Model architecture settings

2. Train the model:
```bash
python train_dualhead.py --cfg configs/yolo11m_dualhead.yaml
```

## 📊 Model Architecture

The system uses a dual-head YOLO architecture:
- **Backbone**: Shared feature extraction
- **PPE Head**: Detection of safety equipment (helmets, vests, head, etc.)
- **Fire Head**: Detection of fire, smoke, and related hazards

## 🗂 Directory Structure

```
unified_safety/
├── configs/               # Configuration files
│   ├── dual_data.yaml    # Dataset configuration
│   └── yolo11m_dualhead.yaml  # Model configuration
├── data/                 # Dataset directory
├── train_dualhead.py    # Training script
└── env.yaml             # Environment configuration
```

## 💡 Usage

### Training

```bash
python train_dualhead.py --cfg configs/yolo11m_dualhead.yaml --data configs/dual_data.yaml
```

### Inference

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/your/video
```

## 📋 Detection Classes

### PPE Detection
- Hard Hats
- Safety Vests
- Safety Gloves
- Safety Goggles
- Safety Boots

### Fire Safety
- Fire
- Smoke

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv8 team for the base architecture
- The manufacturing safety community for dataset contributions
- All contributors and maintainers

## 📧 Contact

- Project Owner: [azimjaan21](https://github.com/azimjaan21)
- Issue Tracker: [Issues](https://github.com/azimjaan21/unified_safety/issues)

---

⭐ Star this repo if you find it helpful!