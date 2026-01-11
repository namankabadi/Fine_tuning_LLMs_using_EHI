# Fine-Tuning LLMs Using Entity Hallucination Index (EHI)

## 📌 Overview

This repository provides an end-to-end framework for analyzing, visualizing, and reducing
**entity-level hallucinations in Large Language Models (LLMs)** using a set of **novel,
entity-centric evaluation metrics**.

The repository contains:
- 📊 Interactive Streamlit visualizations
- 🧠 LLM fine-tuning codebase
- 📈 Metric plots and result snapshots
- 🧪 Entity-aware evaluation beyond traditional ROUGE metrics

The primary goal is to **quantitatively measure factual faithfulness at the entity level**
and enable fine-tuning strategies that reduce hallucination while preserving useful abstraction.

---

## 📊 Streamlit Visualization

The Streamlit application provides:

- Before vs After Fine-Tuning comparison
- Metric-wise line plots (EHI, EF1, PH, OF, NH, LF, EF)
- Combined hallucination metric plots
- Cross-model comparison (Mistral, DistilBART, Flan-T5)
- Average metric tables

### ▶️ Run the App

```bash
streamlit run Streamlit_App_EHI.py

---

## 📸 Results & Visualizations

### 🧪 Streamlit UI Overview
![Streamlit UI](Fine_tuning_LLMs_using_Entity_Hallucination_Index/Results_SnapShots/UI_EHI.png)

---

### 📊 Average Metric Tables

**DistilBART**
![Avg DistilBART](Fine_tuning_LLMs_using_Entity_Hallucination_Index/Results_SnapShots/Avg_Table_DistilBart.png)

**Flan-T5**
![Avg Flan-T5](Fine_tuning_LLMs_using_Entity_Hallucination_Index/Results_SnapShots/Avg_Table_FlanT5.png)

**Mistral**
![Avg Mistral](Fine_tuning_LLMs_using_Entity_Hallucination_Index/Results_SnapShots/Avg_Table_Mistral.png)

---

### 📈 Model-wise EHI & EF1 Line Plots

**DistilBART**
![DistilBART EHI EF1](Fine_tuning_LLMs_using_Entity_Hallucination_Index/Results_SnapShots/DistilBart_EHI_EF1_LinePlot.png)

**Flan-T5**
![Flan-T5 EHI EF1](Fine_tuning_LLMs_using_Entity_Hallucination_Index/Results_SnapShots/EHI_EF1_LinePlot_FlanT5 Model.png)

**Mistral**
![Mistral EHI EF1](Fine_tuning_LLMs_using_Entity_Hallucination_Index/Results_SnapShots/EHI_EF1_LinePlot_Mistral_Model.png)

---

### 🔄 Cross-Model Comparison

![Cross Model EHI](Fine_tuning_LLMs_using_Entity_Hallucination_Index/Results_SnapShots/CrossModel_UI_Plot_EHI.png)

