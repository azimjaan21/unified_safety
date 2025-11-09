# Unified Safety YOLO — Real-Time Industrial PPE + Fire Detection

## 🚧 Overview
The **Unified Safety YOLO** system is a fine-tuned real-time computer vision pipeline designed for **industrial safety monitoring**.  
It unifies **PPE detection** (helmet, vest, head) and **fire detection** into a **single YOLOv11m model**, achieving both high accuracy and high FPS suitable for multi-camera deployments.

This project builds upon multi-task fusion research at **DALAB (CBNU)**, where separate PPE and Fire modules were merged and re-optimized for real-time industrial surveillance.

---

## ⚙️ Model Highlights
- 🧠 **Unified Training:** Combined multi-source datasets (PPE + Fire) with 4 classes  
0 – helmet
1 – vest
2 – head
3 – fire

## 🏭Real Manufacturing Scenarios (Model Performance)


## 🚀Results 
<img width="2400" height="1200" alt="image" src="https://github.com/user-attachments/assets/0bb42f68-802f-41e8-a259-2d5815865c7a" />

  
- 🔄 **Fine-Tuned Backbone:** YOLOv11m optimized for mixed indoor/outdoor industrial scenes  
- ⚡ **High-Speed Inference:** Exported to TensorRT (FP16 / INT8) for deployment on edge GPUs  
- 🎥 **Real-Time Demo:** Supports 8–10 simultaneous camera feeds. 

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

- Azimjon Axtamov, DALAB | Chungbuk National University (CBNU)
- Unified Multimodal Safety YOLO: Real-Time Industrial PPE + Fire Detection
