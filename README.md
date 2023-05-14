# LADE
Learning to Learn Evolutionary Algorithm: A Learnable Differential Evolution

This code is the official implementation of the paper [Learning Adaptive Differential Evolution Algorithm From Optimization Experiences by Policy Gradient](https://ieeexplore.ieee.org/document/9359652).

## Requirements

- Python 3.6
- Torch 1.3.1

## Run 
 
To train and test the deep model on CEC'13 benchmark functions, excute this command:

```sh
$ python main.py
```

## Results

The **trained agent** will be `saved` and the optimization **results** on the test functions are also stored as a `.txt file`. 


## Citation

If you find this repository useful for your work, please cite:

@ARTICLE{LADE,
  author={Liu, Xin and Sun, Jianyong and Zhang, Qingfu and Wang, Zhenkun and Xu, Zongben},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence}, 
  title={Learning to Learn Evolutionary Algorithm: A Learnable Differential Evolution}, 
  year={2023},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TETCI.2023.3251441}
  }
