# OLAF: A Plug-and-Play Framework for Enhanced Multi-object Multi-part Scene Parsing

[Pranav Gupta](https://www.linkedin.com/in/pranav77/)<sup>1</sup>, 
[Rishubh Singh](https://rishubhsingh.github.io/)<sup>2,3</sup>, 
[Pradeep Shenoy](https://sites.google.com/site/pshenoyuw/)<sup>3</sup>, 
[Ravi Kiran Sarvadevabhatla](https://ravika.github.io/)<sup>1</sup>

<br>
<sup>1</sup>International Institute of Information Technology, Hyderabad,  <sup>2</sup>Swiss Federal Institute of Technology (EPFL), <sup>3</sup>Google Research

<br>
ECCV 2024

#### [Project Page](https://olafseg.github.io/) | [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04338.pdf) | [Arxiv](https://arxiv.org/pdf/2411.02858) | 

## Dependencies
Create Conda environment with Python 3.8, PyTorch 1.12.0, and CUDA 10.2 using the following commands:
  ```bash
  conda env create -f ./environment.yml
  conda activate olafseg
  ```
  
### Training & Evaluation
```bash
python run/train.py -dataset pascal-part-58
python run/infer.py -dataset pascal-part-58
```  


## Citation
If you find our methods useful, please cite:
```
@inproceedings{gupta2025olaf,
  title={OLAF: A Plug-and-Play Framework for Enhanced Multi-object Multi-part Scene Parsing},
  author={Gupta, Pranav and Singh, Rishubh and Shenoy, Pradeep and Sarvadevabhatla, Ravi Kiran},
  booktitle={European Conference on Computer Vision},
  pages={178--196},
  year={2025},
  organization={Springer}
}
```
