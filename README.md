# 🚀 RobustOnion (ECCV-26 🎉)<br> [Project](https://ucf-crcv.github.io/RobustOnion/) | [Paper](https://eccv.ecva.net/virtual/2026/poster/5437) | [Results](Benchmark/)  

## Important Note: 
This Paper is a part of my ongoing research, and hence, only the selective code is available. First Please setup GLIP / MMGDINO / GLEE setup (from their original Github); GLIP uses "maskrcnn_benchmark", 

(i.e., maskrcnn_benchmark library, congratulations!!), please upload the above files to your library to enable training GLIP via freezing it entirely, TK0, and NN.

Feel free to raise a GitHub issue if something is not clear. 
If you have your datasets set up, feel free to give it a try or verify performance on Flicker training and COCO evaluation to verify the numbers reported in the paper. 


## Setup 
- [Models](Model_Setup.md) Setup instructions for transformer based models. 
- [Datasets (& New Datsets)](Dataset_Setup.md) Setup instructions for custom datatsets and datasets used in the paper. Convert arbitrary dataset into coco annotations and add it to the model. 
- [Benchmark](Benchmark/) Benchmark csvs used in the paper.
- [Noises](Scripts/noise.py) Simulate noises on images 




## Model Code for RobustOnion (NN+TK0)

** CODE UPLOAD COMING SOON (Before Conference in Sept) ** 

<br> <p align="left">💡 Highlights</p>
</div>
✨ <br/>
</div>






## ✏️ Citation
This Work (Paper 2): Robust onion: Peeling Open Vocab Object Detectors Under Noise 

```bibtex
@InProceedings{robust_onion,
    author="Pathak, Priyank and Karuppasamy, Mukilan and Baranwal, Aaditya and Vyas, Shruti and Rawat, Yogesh S.",
    title="Robust Onion: Peeling Open Vocab Object Detectors Under Noise",
    booktitle="Computer Vision -- ECCV 2026",
    year="2026",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="353--373",
    isbn="978-3-032-37095-2"
}      
```


Paper 1: LR0.FM: Low-Res Benchmark and Improving Robustness for Zero-Shot Classification in Foundation Models [webpage](https://ucf-crcv.github.io/lr0.fm/) | [Github](https://github.com/shyammarjit/LR0.FM) 


```bibtex
@inproceedings{
    pathak2025lrfm,
    title={{ LR0.FM: Low-Res Benchmark and Improving robustness for Zero-Shot Classification in Foundation Models} },
    author={Priyank Pathak and Shyam Marjit and Shruti Vyas and Yogesh S Rawat},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=AsFxRSLtqR}
}
```
