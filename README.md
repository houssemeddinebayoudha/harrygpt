
<div align="center">

# HarryGPT

<img src="images/harryPT.webp" alt="HarryGPT" width="500px"/>

</div>

---

**HarryGPT** is a powerful text generation model based on the **GPT** (Generative Pre-trained Transformer) architecture. It follows a **decoder-only** approach, which is commonly used in modern text generation models for creating natural and coherent language outputs. 

In building **HarryGPT**, I began by implementing the core concepts from the 2017 research paper, **"Attention is All You Need"** by Google, using **PyTorch**. After successfully reproducing the transformer, I made essential modifications to adapt the model for text generation tasks, as shown in the figure below.

<div align="center">
<hr width="900px">
</div>

<div align="center"> 
<img src="images/gptArchitecture.png" alt="GPT Architecture" width="1000px"/>
</div>

To train **HarryGPT**, I utilized the publicly available **Harry Potter** book series, allowing the model to generate text that mimics the style and tone of the Harry Potter universe.

The model also uses **Byte-Pair Encoding (BPE) tokenization** for handling input text efficiently.

---

### About This Repository

This repository hosts the **Streamlit app** version of the HarryGPT model. You can test the model live using the following link: [HarryGPT Live Demo](https://wizardgpt.live/).

Alternatively, you can also clone this repository and run the app locally. The project uses **Git LFS (Large File Storage)** for managing large files, so make sure to install and initialize Git LFS before cloning the repository:

1. Install Git LFS from [Git LFS website](https://git-lfs.github.com/):
   ```bash
   git lfs install
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/houssemeddinebayoudha/harrygpt.git
   ```

3. To run the app locally, you can use **Docker**. Simply build the Docker container and start the app:
   ```bash
   docker build -t harrygpt-app .
   docker run -p 8501:8501 harrygpt-app
   ```

---

### Key Files

- **[model.py](./model.py)**: Contains the HarryGPT text generation model implementation.
- **[app.py](./app.py)**: Contains the streamlit app.

Feel free to explore the repository, clone it, and try out the model locally or through the live demo.

---
### References

1. **Karpathy's NanoGPT**: [Karpathy NanoGPT GitHub](https://github.com/karpathy/nanoGPT)
2. **Google's "Attention is All You Need" Paper**: [Attention is All You Need](https://arxiv.org/abs/1706.03762)
