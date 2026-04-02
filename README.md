
# 🛡️ Safe-Guard AI: Automated Industrial Safety Monitoring

[](https://safe-guard-ai-ybsy3eegg5v7yxqynrktn5.streamlit.app/)
[](https://opensource.org/licenses/MIT)
[](https://www.python.org/downloads/)

**Safe-Guard AI** ek advanced Computer Vision solution hai jo industrial environments (construction sites, factories, aur warehouses) mein safety protocols ki monitoring ko automate karta hai. Yeh system real-time mein workers ke PPE (Personal Protective Equipment) ko detect karta hai aur violations par fori action leta hai.

-----

## 🚀 Live Demo

Project ko live dekhne ke liye yahan click karein: [Safe-Guard AI Live App](https://safe-guard-ai-ybsy3eegg5v7yxqynrktn5.streamlit.app/)

-----

## 📝 Project Overview

Industrial sectors mein accidents ki sabse bari wajah safety gear (Helmet/Vest) ka istemal na karna hai. **Safe-Guard AI** is masle ko hal karne ke liye AI aur Automation ka istemal karta hai taaki 24/7 monitoring baghair kisi insani thakawat ke mumkin ho sake.

### 🚩 Problem Statement

Manual monitoring bohat mushkil aur slow hoti hai. Jab tak supervisor ko pata chalta hai ke kisi worker ne helmet nahi pehna, tab tak hadsa ho chuka hota hai. Is system ka maqsad is gap ko khatam karna hai.

-----

## ✨ Key Features

  * **Real-time PPE Detection:** YOLOv8 ka istemal karte hue Helmet aur Safety Vests ki tez-raftaar detection.
  * **Face-Linked Violations:** Agar koi rule torta hai, toh DeepFace ke zariye worker ki pehchan (Name & ID) database se nikal li jati hai.
  * **Interactive Analytics Dashboard:**
      * **Live Metrics:** Total scans aur compliance rate ke real-time numbers.
      * **Safety Calendar:** Mahine bhar ki safety performance ka visual heatmap (Green/Red dots).
      * **Trend Graphs:** Hourly aur daily violations ka graph.
  * **Instant Email Alerts:** Pipedream/N8N integration ke zariye management ko fori notification.
  * **Worker Management:** Naye workers ko camera ya photo upload ke zariye register karne ki sahulat.
  * **Batch Analysis:** Purani images ko ek saath upload karke audit report generate karna.

-----

## 🛠️ Tech Stack

| Category | Tools / Technologies |
| :--- | :--- |
| **Language** | Python 3.9+ |
| **AI Models** | YOLOv8 (Object Detection), DeepFace (Recognition) |
| **Backend API** | FastAPI (Hosted on Hugging Face Spaces) |
| **Frontend UI** | Streamlit |
| **Database** | SQLite3 |
| **Automation** | Pipedream / N8N (Email Webhooks) |
| **Visualization** | Plotly, Matplotlib, Pandas |

-----

## ⚙️ Installation & Setup

1.  **Repo Clone Karein:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/safe-guard-ai.git
    cd safe-guard-ai
    ```

2.  **Dependencies Install Karein:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Application Run Karein:**

    ```bash
    streamlit run app.py
    ```

-----

## 📂 Project Structure

```text
├── app.py              # Main Streamlit Frontend & Logic
├── worker_faces/       # Registered workers ki photos ka folder
├── safety_violations.db # SQLite database (Logs aur violations)
├── requirements.txt    # Zaruri Python libraries ki list
└── README.md           # Project Documentation
```

-----

## 🤝 Connect with Me

Agar aapko is project ke bare mein koi sawal ho ya aap collaboration chahte hon:

  * **Developer:** Shumas Kashif Chohan
  * **LinkedIn:** [Shumas Kashif Chohan](https://www.linkedin.com/in/shumas-kashif-chohan-54b51830b?utm_source=share_via&utm_content=profile&utm_medium=member_android)
  * **Portfolio:** [Live Project Link](https://safe-guard-ai-ybsy3eegg5v7yxqynrktn5.streamlit.app/)

-----

## 📜 License

Yeh project MIT License ke tehat release kiya gaya hai.
