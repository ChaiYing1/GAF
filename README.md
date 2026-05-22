# GAF: Gaussian Action Field as a 4D Representation for Dynamic World Modeling in Robotic Manipulation

[![ICRA 2026](https://img.shields.io/badge/Accepted%20to-ICRA%202026-red)](https://ieee-ras.org/conferences-workshops/future-of-icra)

<p align="center">
  <a href="">
    <img src="https://github.com/ChaiYing1/GAF.github.io/blob/main/static/images/overview.jpg" alt="Teaser" width="100%">
  </a>
</p>

## 📄 Paper

- **Paper (PDF)**: [arXiv:2506.14135](https://arxiv.org/pdf/2506.14135)
- **arXiv**: [https://arxiv.org/abs/2506.14135](https://arxiv.org/abs/2506.14135)
- **Status**: Accepted to ICRA 2026

## 🎥 Demo Video

See the [project website](https://chaiying1.github.io/projects/GAF/) for the full demo video.

## 🏗️ Architecture

GAF provides three interrelated outputs:
- **Current Gaussian**: GAF reconstructs 3D Gaussian Pointclouds at current timestep. 
- **Future Gaussian**: GAF predicts future 3D Gaussian Pointclouds by motion attributes. 
- **Init action**: The init action is estimated through point cloud matching between the current and future manipulation-related Gaussian.

Both the high-level module and adaptive corrector are implemented as causal transformers that leverage historical scanning sequences for more informed decision-making.

## 📁 Repository Structure

The project is logically divided into distinct modules for 4D reconstruction and action refinement:
* `gaf/`: Contains the code for training the core Gaussian Action Field model for 4D scene reconstruction and motion prediction.
* `refine/`: Contains the diffusion-based action-vision-aligned denoising framework for action refinement.
* `eval/`: Contains scripts for evaluating the trained models in simulated environments (Stay tuned!).

## ⚙️ Environment Setup

The environment for this project is a hybrid setup. We recommend using **BF16 precision** for optimal training performance and memory management.

Our code relies on Python 3.10+, and is developed based on PyTorch 2.1.2 and CUDA 11.8, but it should work with higher Pytorch/CUDA versions as well.

1. Clone NoPoSplat.
```bash
git clone https://github.com/ChaiYing1/GAF
cd GAF/gaf
```

2. Create the environment, here we show an example using conda.
```bash
conda create -y -n gaf python=3.10
conda activate gaf
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

cd ../refine/rendiff
conda env update -n gaf -f environment.yml
```

3. Install PyRep and RLbench by following the instructions in the https://github.com/stepjam/PyRep
and https://github.com/stepjam/RLBench

## 🚀 Training Pipeline

The training process is split into two sequential phases: training the 4D GAF representation, and training the refinement diffusion model.

### Phase 1: Train the GAF Model
Navigate to the `gaf` directory to train the 4D reconstruction model. This model learns the time-varying scene geometry and outputs the current scene, future frame predictions, and init action estimations.

```bash
python -m src.main +experiment=gaf wandb.mode=online wandb.name=gaf
```

### Phase 2: Train the Refinement Model
Once the GAF model is trained, use it to process your dataset. The GAF model will generate the corresponding `init action` and multi-view RGB images. These outputs serve as the Actionable Multiview RGB Guidance for the denoising framework.

Navigate to the `refine` directory and run the following command to train the refinement model:

```bash
python3 -m rendiff.train \
    --run_name='LIFT_LID' \
    --datadir='/path/to/datadir/task_name' \
    --num_demos=25
```

## 📊 Evaluation and Testing

To test the complete closed-loop framework on specific manipulation tasks, navigate to the `eval` directory. The evaluation script will iteratively query the GAF model and the refinement network until the manipulation task completes. 


## 📝 Citation

If you find this work useful in your research, please cite:

```bibtex
@misc{chai2025gafgaussianactionfield,
      title={GAF: Gaussian Action Field as a 4D Representation for Dynamic World Modeling in Robotic Manipulation}, 
      author={Ying Chai and Litao Deng and Ruizhi Shao and Jiajun Zhang and Kangchen Lv and Liangjun Xing and Xiang Li and Hongwen Zhang and Yebin Liu},
      year={2025},
      eprint={2506.14135},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.14135}, 
}
```

## 🌐 Project Website

Visit our project website: [[https://chaiying1.github.io/GAF.github.io/project_page/](https://chaiying1.github.io/projects/GAF/)]

This website is deployed and maintained by [Ying Chai].

## 📄 License

This website is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

## 🙏 Acknowledgments

This website borrows the source code of [this website](https://nerfies.github.io/). We sincerely thank [Keunhong Park](https://keunhong.com/) for developing and open-sourcing this template.

## 📧 Contact

For questions or inquiries, please contact:
- **Ying Chai** : chaiy25@mails.tsinghua.edu.cn

---

**Note**: This work was supported by the National Natural Science Foundation of China (NSFC) No.62125107.
