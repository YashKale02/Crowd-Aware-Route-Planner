# Crowd-Aware Route Planner using YOLOv8 and A* Algorithm

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8x-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b)

## 📜 Overview
This project is a **Crowd-Aware Route Planner** that fuses deep learning-based perception with heuristic pathfinding algorithms. Designed for dynamic, human-centric environments, the system detects pedestrians in video frames and computes an optimal, safe navigation path that avoids high-density crowd clusters.

Unlike traditional planners that rely on static obstacle maps, this system generates a dynamic **crowd density heat map** and applies safety buffering to ensure sufficient clearance between the route and detected humans.

## 🚀 Features
* **Perception:** Uses **YOLOv8x** to detect humans and obstacles with high semantic understanding.
* **Crowd Mapping:** Converts bounding boxes into a weighted **crowd density grid** rather than a simple binary obstacle map.
* **Safety Buffering:** Implements **Morphological Dilation** to "inflate" obstacles, creating a safety buffer zone around people.
* **Pathfinding:** Utilizes the **A* (A-Star) Algorithm** with an Octile heuristic to find the optimal path balancing distance vs. crowd density.
* **Visualization:** Features a **Streamlit** dashboard and **OpenCV** rendering to display the raw path, smoothed path, and cost map.

## 🏗️ System Architecture
The system operates in five distinct layers:
1.  **Input Layer:** Captures static images or frames.
2.  **Detection Layer:** YOLOv8x processes frames to output bounding boxes for class 'person'.
3.  **Mapping Layer:** Generates a grid (GRID_SIZE × GRID_SIZE), maps centroids, and applies dilation.
4.  **Planning Layer:** Computes the route using A* based on the normalized cost map.
5.  **Visualization Layer:** Renders the final output with the smoothed path.

## 🧠 Algorithm Summary
The system uses the **A* Algorithm** to minimize the total cost function $f(n) = g(n) + h(n)$.

* **Cost (g(n))**: Incorporates a weighted crowd penalty (CROWD_WEIGHT × cost_grid[n]) to actively discourage paths near detected people.
* **Heuristic (h(n))**: Uses Octile distance to accurately estimate diagonal movement costs.

This hybrid approach ensures the agent selects paths that optimally balance physical shortness with safety clearance.

## 🛠️ Tech Stack
| Component | Tool/Library | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.10 | Core logic |
| **Detection** | Ultralytics YOLOv8 | Object detection model |
| **Vision** | OpenCV | Image processing & Dilation |
| **Math** | NumPy & SciPy | Grid manipulation & inflation |
| **UI** | Streamlit | Interactive dashboard |

## 💻 Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/Crowd-Aware-Route-Planner.git](https://github.com/your-username/Crowd-Aware-Route-Planner.git)
    cd Crowd-Aware-Route-Planner
    ```

2.  **Install dependencies**
    ```bash
    pip install ultralytics opencv-python numpy scipy streamlit
    ```

3.  **Run the application**
    ```bash
    streamlit run app.py
    ```


## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
