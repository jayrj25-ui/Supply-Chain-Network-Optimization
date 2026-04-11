# 🚀 Supply Chain Resilience Optimizer

### AI-Driven Pipeline using GNN + GRU · Disruption Simulation · LP Optimization

---

## 📌 Overview

Modern supply chains are highly vulnerable to disruptions such as factory shutdowns, port closures, and demand fluctuations.
This project presents an **end-to-end intelligent system** that combines **Graph Neural Networks and Optimization techniques** to enhance supply chain resilience.

🔹 Predict demand across the network
🔹 Simulate real-world disruptions
🔹 Optimally re-route supply at minimum cost

---

## 🧠 Methodology Pipeline

```
Step 1: GNN (Graph + GRU)
        ↓  Demand Forecasting
Step 2: Disruption Simulation
        ↓  Network Stress Testing
Step 3: Optimization (LP / Min-Cost Flow)
        ↓  Cost-Optimal Re-routing
```

---

## ⚙️ Key Components

### 🔹 Demand Forecasting (GNN + GRU)

* Models supply chain as a **graph structure**
* Uses **2-layer GCN** for spatial dependency learning
* Uses **GRU** to capture temporal demand patterns
* Outputs **node-wise demand predictions**

---

### 🔹 Disruption Simulation

* Simulates failures by removing nodes/edges
* Represents:

  * Factory shutdowns
  * Logistics disruptions
  * Infrastructure failures
* Evaluates **network resilience metrics**

---

### 🔹 Optimization (Min-Cost Flow LP)

* Formulates a **Linear Programming model**
* Uses **PuLP (CBC Solver)**
* Ensures:

  * Flow conservation
  * Capacity constraints
  * Minimum transportation cost
* Outputs **optimal supply routing strategy**

---

## 📊 Results & Insights

* ✔ Accurate demand forecasting using GNN+GRU
* ✔ Successful disruption simulation (20% node removal)
* ✔ Optimal re-routing achieved using LP solver
* ✔ Demonstrates **resilient supply chain behavior under stress**

| Metric    | Status              |
| --------- | ------------------- |
| RMSE      | Available in output |
| MAE       | Available in output |
| R²        | Available in output |
| LP Solver | Optimal             |

---

---

---

## 🛠️ Tech Stack

| Domain           | Tools             |
| ---------------- | ----------------- |
| Machine Learning | PyTorch           |
| Graph Modeling   | NetworkX          |
| Optimization     | PuLP (CBC Solver) |
| Data Processing  | Pandas, NumPy     |
| Evaluation       | Scikit-learn      |
| Visualization    | Matplotlib        |

---


## 📂 Project Structure

```
Supply-Chain-Network-Optimization/
├── main.py
├── requirements.txt
├── README.md
├── report/
│   └── supply_chain_report.docx
├── demo/
│   └── demo.html
```

---

## 📈 Future Improvements

* Multi-product supply chain modeling
* Real-time dashboard (Streamlit / Dash)
* Stochastic optimization under uncertainty
* Edge disruption modeling

---

## 👨‍💻 Author

**Jay Rathod**
M.Tech – Industrial Engineering & Management (IIT Kharagpur)

---

## ⭐ If you found this useful, consider giving a star!
