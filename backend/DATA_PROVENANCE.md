# Remote Sensing Dataset Provenance Documentation
**SIH 2026 Problem Statement 26167 | Team Vyomix**

This document establishes the official data provenance, licenses, sensor modalities, and ground-truth structures for all Earth Observation datasets integrated into SatQuery AI.

---

## 1. BigEarthNet.txt (Vision-Language Remote Sensing Benchmark)
- **Official Source**: BIFOLD / Technische Universität Berlin & German Aerospace Center (DLR)
- **Repository**: [https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt)
- **Primary Reference**: *"BigEarthNet.txt: A Large-Scale Vision-Language Dataset for Remote Sensing"* (2024)
- **License**: Community Data License Agreement – Permissive (CDLA-Permissive-1.0)
- **Sensors & Modalities**:
  - **Sentinel-2 MSI**: Optical Multispectral (B02 Blue, B03 Green, B04 Red, B08 NIR at 10m resolution, plus 20m red-edge and SWIR bands).
  - **Sentinel-1 SAR**: Dual-polarization C-band backscatter (VV + VH in GRD format at 10m resolution).
- **Annotation Format**: Parquet index (`BigEarthNet.txt.parquet`) with instruction questions (`input`), reference ground-truth answers (`output`), task categories (binary classification, multi-class land cover, scene description), and Corine Land Cover (CLC-19 / CLC-43) taxonomy.
- **Intended Purpose**: VLM domain adaptation, Sentinel-2 spectral land cover VQA, and cross-modal optical-SAR understanding.

---

## 2. VRSBench (High-Resolution Earth Observation VLM Benchmark)
- **Official Source**: State Key Laboratory of Virtual Reality Technology and Systems, Beihang University
- **Repository**: [https://huggingface.co/datasets/xiang709/VRSBench](https://huggingface.co/datasets/xiang709/VRSBench) & [https://github.com/lx709/VRSBench](https://github.com/lx709/VRSBench)
- **Primary Reference**: *"VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding"* (NeurIPS 2024 Track on Datasets and Benchmarks)
- **License**: Creative Commons Attribution 4.0 International (CC-BY 4.0)
- **Sensors & Modalities**: Sub-meter high-resolution RGB aerial and satellite imagery (Google Earth, DOTA, NWPU-RESISC45).
- **Annotation Statistics**:
  - **29,614 images** with human-verified annotations.
  - **123,221 Question-Answer Pairs** (`VRSBench_EVAL_vqa.json`, `VRSBench_train.json`).
  - **29,614 Detailed Scene Captions** (`VRSBench_EVAL_Cap.json`).
  - **52,472 Visual Grounding References** with exact bounding boxes (`VRSBench_EVAL_referring.json`).
- **Intended Purpose**: Text-guided referring expression localization (bounding box grounding), dense scene captioning, and open-vocabulary VQA evaluation.

---

## 3. RSVQA (Remote Sensing Visual Question Answering)
- **Official Source**: Sylvain Lobry, Diego Marcos, Jesse Murray, Devis Tuia (University of Lausanne / Wageningen University)
- **Repository**: [https://rsvqa.sylvainlobry.com/](https://rsvqa.sylvainlobry.com/)
- **Primary Reference**: *"RSVQA: Visual Question Answering for High-Resolution Remote Sensing Data"* (IEEE TGRS 2020)
- **License**: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)
- **Sensors & Modalities**: Sentinel-2 (RSVQA-LR, 10m) and high-resolution aerial imagery (RSVQA-HR, 15cm).
- **Annotation Format**: Questions covering presence/absence ("Is there a..."), comparison, area count, and rural/urban land use.
- **Intended Purpose**: Standardized zero-shot and adapted accuracy benchmark for quantitative VQA evaluation.
