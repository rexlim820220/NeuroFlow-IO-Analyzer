# PyEdge-Lab-Integrator 🚀

## 📌 專案簡介
本專案是一個基於 Python 的模組化研發實驗平台，旨在探索邊緣運算環境下的影像處理、硬體控制通訊及高效能運算架構。本平台採用物件導向（OOP）設計與 MVC 概念，提供直觀的 GUI 介面進行多項技術指標測試。

<img width="576" height="661" alt="image" src="https://github.com/user-attachments/assets/109afae6-1bbb-45f6-ad76-a2d63b721925" />

## 🛠️ 四大核心研究課題

### 1. OpenCV 影像處理引擎
- **技術重點**：封裝影像濾鏡演算法，並將 OpenCV 影像矩陣動態映射至 Tkinter 介面。
- **功能實作**：包含灰階處理、高斯模糊及邊緣檢測等基礎演算法之整合。

### 2. Edge-AI YOLO 模型效能評估 (WIP)
- **技術重點**：在資源受限環境（如低功耗嵌入式設備）下，針對不同規模的 YOLO 模型進行精準度與推論速度權衡。
- **目標**：優化模型參數，實現即時偵測。

### 3. Python 進階技術與工業級 I/O 串接
- **技術重點**：
  - **Decorator**：用於統一處理通訊異常與 Log 記錄。
  - **Generator/Iterator**：應用於流暢的影像採集與非同步數據讀取。
- **硬體串接**：支援 **PLC (Modbus)** 與 **RS232 (Serial)** 協議，實現軟硬體同步監控。

### 4. 並行運算效能實驗室 (Concurrency Lab)
- **技術重點**：深入比較 **Multi-threading** 與 **Multi-processing** 的差異。
- **實驗對象**：
  - **CPU-Bound**：測試 Python GIL (Global Interpreter Lock) 對計算密集型任務的限制。
  - **I/O-Bound**：模擬硬體通訊等待時間，驗證並行處理對反應速度的提升。

---

## 📂 專案架構
```text
PyEdge-Lab-Integrator/
├── main.py                # 程式啟動入口
├── app_controller.py      # 主視窗路由與頁面管理
├── views/                 # UI 介面層 (各類別繼承自 BasePage)
│   ├── base_view.py
│   └── opencv_view.py ... 
├── core/                  # 核心邏輯層 (OpenCV 處理、PLC 通訊)
│   └── performance_lab.py
└── assets/                # 模型權重與實驗數據
