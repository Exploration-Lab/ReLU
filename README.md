# ReLU

This is the repository that contains source code for the [ReLU website](https://exploration-lab.github.io/ReLU/). 

If you find ReLU useful for your research please cite:
```
@inproceedings{joshi-etal-2024-towards,
    title = "Towards Robust Evaluation of Unlearning in {LLM}s via Data Transformations",
    author = "Joshi, Abhinav  and
      Saha, Shaswati  and
      Shukla, Divyaksh  and
      Vema, Sriram  and
      Jhamtani, Harsh  and
      Gaur, Manas  and
      Modi, Ashutosh",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.706",
    pages = "12100--12119",
    abstract = "Large Language Models (LLMs) have shown to be a great success in a wide range of applications ranging from regular NLP-based use cases to AI agents. LLMs have been trained on a vast corpus of texts from various sources; despite the best efforts during the data pre-processing stage while training the LLMs, they may pick some undesirable information such as personally identifiable information (PII). Consequently, in recent times research in the area of Machine Unlearning (MUL) has become active, the main idea is to force LLMs to forget (unlearn) certain information (e.g., PII) without suffering from performance loss on regular tasks. In this work, we examine the robustness of the existing MUL techniques for their ability to enable leakage-proof forgetting in LLMs. In particular, we examine the effect of data transformation on forgetting, i.e., is an unlearned LLM able to recall forgotten information if there is a change in the format of the input? Our findings on the TOFU dataset highlight the necessity of using diverse data formats to quantify unlearning in LLMs more reliably.",
}
```

