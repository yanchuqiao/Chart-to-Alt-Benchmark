#  **Chart-to-Alt: Training & Evaluation**

# Overview
This repository contains the training and evaluation workflows for the Chart-to-Alt dataset.
The Chart-to-Alt dataset is the first large-scale, real-world dataset for scientific chart accessibility, consisting of:

10,000 pairs of realistic scientific chart images and alternative text (alt text).
Charts collected from 3 different digital libraries.
20 topical keywords covering a wide variety of chart types.
Alt text manually created by 10 trained annotators using a 4-level semantic model to ensure comprehensive and structured descriptions.
🔗 Dataset: [yanchuqiao/Chart-to-Alt_v2](https://huggingface.co/datasets/yanchuqiao/Chart-to-Alt_v2)


# Structure
This repository includes four ZIP archives, each containing scripts and results for fine-tuning and inference on the Chart-to-Alt_v2 dataset.
Fine-Tuning & Inference Methods
The scripts are adapted from the following repositories:
| Method            | Source Repository                                                                           |
| ----------------- | ------------------------------------------------------------------------------------------- |
| **UniChart**      | [vis-nlp/UniChart](https://github.com/vis-nlp/UniChart)                                     |
| **TinyChart**     | [X-PLUG/mPLUG-DocOwl/TinyChart](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/TinyChart) |
| **ChartGemma**    | [vis-nlp/ChartGemma](https://github.com/vis-nlp/ChartGemma)                                 |
| **ChartInstruct** | [vis-nlp/ChartInstruct](https://github.com/vis-nlp/ChartInstruct)                           |

# Evaluation
Evaluation scripts are organized separately and support:
* 1 prompt evaluation.
* 2 prompts evaluation.
  
These scripts apply for all four methods. 
