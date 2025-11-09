- 🔄 **Fine-Tuned Backbone:** YOLOv11m optimized for mixed indoor/outdoor industrial scenes  
- ⚡ **High-Speed Inference:** Exported to TensorRT (FP16 / INT8) for deployment on edge GPUs  
- 🎥 **Real-Time Demo:** Supports 8–10 simultaneous camera feeds on TITAN RTX  

---

## 🧩 Dataset Composition
| Split | Images | Boxes | Notes |
|-------|---------|--------|-------|
| **Train** | 1,751 | 9,163 | Unified industrial scenes (factory, lab, warehouse) |
| **Val** | 198 | 805 | Clean validation across mixed conditions |
| **Total** | 1,949 | 9,968 | 4 classes (Helmet, Vest, Head, Fire) |

---

## 🚀 Performance (TITAN RTX, 24 GB)
| Precision Mode | Framework | FPS (avg) | Relative Speed | Visual Chart |
|----------------|------------|-----------|----------------|---------------|
| **FP32** | PyTorch | 45 | 1.0× | ▓▓▓ |
| **FP16** | TensorRT | 68 | 1.5× | ▓▓▓▓▓ |
| **INT8** | TensorRT | 92 | 2.0× | ▓▓▓▓▓▓ |

🟢 *FP16 delivers the best balance between speed and accuracy for most industrial settings.*

---

## 🧱 Applications
- 🏭 Real-time industrial safety monitoring  
- 🔥 Fire & hazard early detection in factories  
- 🧍 PPE compliance verification (helmet, vest, head)  
- 🎯 Edge deployment on Jetson / TITAN RTX / RTX 6000  

---

## 🧠 Citation / Acknowledgment
If you use this work, please acknowledge:

- Azimjon Axtamov, DALAB @ Chungbuk National University (CBNU)
- Unified Multimodal Safety YOLO: Real-Time Industrial PPE + Fire Detection