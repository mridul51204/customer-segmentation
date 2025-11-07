# 🧩 Customer Segmentation Dashboard

A data-driven **Customer Segmentation Dashboard** designed to identify customer groups based on purchasing patterns and behavioral attributes. This project leverages **K-Means clustering** and exploratory data analysis to help businesses understand their customer base and make informed marketing or product decisions.

🔗 **Live App:** [Customer Segmentation Dashboard](https://customer-segmentation-82jjwo9sg5hrkm9w6iggqo.streamlit.app/)

---

## 📘 Overview
Understanding customer diversity is essential for personalization, marketing optimization, and product targeting. This project segments customers into distinct clusters using unsupervised machine learning — highlighting **key behavioral insights, spending profiles, and demographics** to support better business strategy.

---

## 🧠 Key Features
- ⚙️ **Automated Data Preprocessing:** Handles missing values and outliers.  
- 📊 **Feature Engineering:** Creates derived variables for better cluster separation.  
- 🧮 **K-Means Clustering:** Groups customers based on multi-dimensional attributes.  
- 🔍 **Insight Generation:** Extracts actionable insights for each cluster.  
- 📈 **Interactive Visuals:** Displays segmentation results and metrics in an easy-to-understand format.  
- 💾 **Modular Codebase:** Organized into reusable scripts for scalability and maintenance.  

---

## 🏗️ Project Structure
```bash
customer-segmentation/
│
├── .streamlit/
│   └── config.toml               # Streamlit configuration (theme, layout, etc.)
│
├── app/
│   └── app.py                    # Main app entry point
│
├── data/
│   ├── raw/                      # Unprocessed input data
│   └── processed/                # Cleaned & transformed data
│
├── src/                          # Core logic and utilities
│   ├── __init__.py
│   ├── clustering.py             # K-Means model training and evaluation
│   ├── data_prep.py              # Data cleaning and preprocessing
│   ├── feature_engineering.py    # Derived feature creation
│   └── insights.py               # Cluster-level insights and metrics
│
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python version for deployment
├── .gitignore
└── README.md                     # Project documentation
```

---

## 🧩 Workflow
1. **Data Ingestion:** Raw data imported into `/data/raw/`  
2. **Preprocessing:** Cleaned and normalized using `data_prep.py`  
3. **Feature Engineering:** Enriched with calculated metrics (`feature_engineering.py`)  
4. **Clustering:** K-Means applied via `clustering.py`  
5. **Insights:** Cluster summaries generated in `insights.py`  
6. **App Display:** Visualized interactively through the live dashboard  

---

## ⚙️ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/mridul51204/customer-segmentation.git
cd customer-segmentation
```
### 2️⃣ Create a Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate        # On Windows
source venv/bin/activate     # On macOS/Linux
```
### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Run the Application
```bash
streamlit run app/app.py
```

---

## 🧮 Tech Stack
- **Language:** Python 3.11+  
- **Core Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn  
- **Visualization:** Plotly / Matplotlib  
- **Deployment:** Streamlit Cloud  

---

## 📈 Example Output
*(Add screenshots or charts here once generated)*  

![Cluster Distribution](link-to-cluster-image)  
![Spending Behavior](link-to-behavior-image)  
![Insights Dashboard](link-to-dashboard-image)  

---

## 🧾 License
This project is open-source and available for educational and non-commercial use.

---

## 👨‍💻 Author
**Mridul Grover**  
📧 [GitHub Profile](https://github.com/mridul51204)

---

### ⭐ If you found this project helpful, don’t forget to star the repo!
