# AI Engineering Demonstrations

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?logo=huggingface&logoColor=000)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?logo=langchain&logoColor=white)
![IBM Watson](https://img.shields.io/badge/IBM-WatsonX-054ADA?logo=ibm&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive portfolio of AI/ML implementations from IBM AI Engineering and RAG & Agentic AI Professional Certifications. Features hands-on projects demonstrating neural networks, transformer architectures, parameter-efficient fine-tuning, retrieval-augmented generation, and multi-agent systems.

## 🎯 Overview

This repository showcases practical implementations across the AI/ML stack, from foundational deep learning architectures to cutting-edge agentic systems. Each project includes complete code, documentation, and demonstrates production-ready patterns.

## ⭐ Featured Projects

Several standout projects have been extracted into dedicated repositories with enhanced documentation:

- 🔗 [**german-english-transformer-nmt**](https://github.com/deepdmk/german-english-transformer-nmt) - Transformer architecture built from scratch for neural machine translation
- 🔗 [**lora-instruction-finetuning**](https://github.com/deepdmk/lora-instruction-finetuning) - Parameter-efficient instruction tuning for code generation
- 🔗 [**dpo-preference-alignment**](https://github.com/deepdmk/dpo-preference-alignment) - Human preference alignment using Direct Preference Optimization
- 🔗 [**beeai-agentic-framework-demos**](https://github.com/deepdmk/beeai-agentic-framework-demos) - Multi-agent orchestration and coordination patterns
- 🔗 [**youtube-rag-qa-application**](https://github.com/deepdmk/youtube-rag-qa-application) - Production RAG system with interactive web interface

## 📚 Projects

### Deep Learning & Neural Networks

#### [Seq2Seq Transformer NMT](./seq2seq-transformer-nmt)
Complete German-to-English neural machine translation system built from scratch.
- **Tech Stack:** PyTorch, Transformers, Multi-head Attention
- **Features:** Training pipeline, BLEU evaluation, PDF translation
- **Highlights:** Encoder-decoder architecture, positional encoding, greedy decoding

#### [Fashion MNIST Classifier CNN](./fashion-mnist-classifier-cnn)
Convolutional neural network for fashion item classification.
- **Tech Stack:** PyTorch, CNNs
- **Dataset:** Fashion MNIST (70,000 images)
- **Highlights:** Multi-layer CNN architecture, data augmentation

#### [Breast Cancer Classification](./breast-cancer-classification)
Binary classification using neural networks for medical diagnosis.
- **Tech Stack:** PyTorch, sklearn
- **Dataset:** Wisconsin Breast Cancer Dataset
- **Highlights:** Feature engineering, model evaluation metrics

#### [Aircraft Damage Classification](./aircraft-damage-classification)
Transfer learning for aircraft damage detection.
- **Tech Stack:** PyTorch, VGG16
- **Highlights:** Transfer learning, computer vision

#### [Waste Classification](./waste-classification-transfer-learning)
Environmental AI for waste sorting and recycling optimization.
- **Tech Stack:** PyTorch, Transfer Learning
- **Highlights:** Multi-class classification, real-world application

### Model Fine-Tuning & Optimization

#### [LoRA Instruction Fine-Tuning](./lora_instruction_fine-tuning)
Parameter-efficient instruction tuning for code generation tasks.
- **Tech Stack:** PEFT, LoRA, Hugging Face Transformers
- **Model:** OPT-350M
- **Dataset:** CodeAlpaca-20k
- **Highlights:** Trains <1% of parameters, SACREBLEU evaluation

#### [DPO Fine-Tuning with LoRA](./dpo_fine_tuning_gpt2_lora)
Human preference alignment using Direct Preference Optimization.
- **Tech Stack:** DPO, LoRA, TRL, Hugging Face
- **Model:** GPT-2
- **Dataset:** UltraFeedback Binarized
- **Highlights:** Alignment without reward models, 4-bit quantization

### Retrieval-Augmented Generation (RAG)

#### [LangChain RAG Search](./rag-langchain-search)
Comprehensive demonstration of RAG retrieval strategies.
- **Tech Stack:** LangChain, ChromaDB, IBM WatsonX
- **Features:** 6 retrieval strategies (similarity, MMR, multi-query, self-querying, parent document)
- **Highlights:** Vector similarity search, metadata filtering, context-rich retrieval

#### [YouTube RAG Q&A Bot](./youtube_langchain_rag_agent)
Production RAG application with interactive web interface.
- **Tech Stack:** LangChain, FAISS, Gradio, IBM WatsonX
- **Features:** Transcript extraction, video summarization, semantic Q&A
- **Highlights:** End-to-end application, production UI

### Agentic AI & Multi-Agent Systems

#### [BeeAI Multi-Agent System](./beeai_multi-agent-travel-system)
Progressive demonstrations of AI agent capabilities from basic chat to multi-agent coordination.
- **Tech Stack:** BeeAI Framework, IBM WatsonX
- **Features:** 12-stage progression covering tools, execution control, custom tools, multi-agent orchestration
- **Highlights:** Human-in-the-loop workflows, agent handoff patterns, production-ready architectures

#### [CrewAI Research & Writing System](./crewai_multi_agent_research_write)
Autonomous multi-agent system for research and content creation.
- **Tech Stack:** CrewAI, LLaMA 3.3 70B, SerperDev API
- **Features:** Role-based specialization, sequential workflows, web search integration
- **Highlights:** Research analyst + content writer coordination

### Multimodal AI

#### [Fashion Image Assistant](./fashion-image-assistant)
AI-powered fashion recommendation system using vision and language models.
- **Tech Stack:** OpenAI GPT-4 Vision, Image Processing
- **Highlights:** Multimodal AI, product recommendations

## 🛠️ Technology Stack

**Deep Learning Frameworks:**
- PyTorch 2.x
- Hugging Face Transformers, PEFT, TRL

**LLM & Agent Frameworks:**
- LangChain, LangGraph
- CrewAI, BeeAI Framework

**Vector Databases:**
- ChromaDB, FAISS

**LLM Providers:**
- IBM WatsonX AI (Granite, LLaMA, Mistral)
- OpenAI (GPT-4, GPT-3.5)

**Additional Tools:**
- Gradio (web interfaces)
- spaCy, NLTK (NLP)
- Pandas, NumPy (data processing)

## 📋 Prerequisites

Each project has its own requirements, but common dependencies include:

```bash
# Core ML/DL
pip install torch torchvision transformers datasets

# LLM & Agent Frameworks
pip install langchain langchain-community langchain-ibm
pip install crewai beeai-framework

# Vector Stores & RAG
pip install chromadb faiss-cpu

# Fine-tuning & Optimization
pip install peft trl bitsandbytes accelerate

# Utilities
pip install gradio jupyter matplotlib pandas
```

See individual project directories for specific requirements.

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/deepdmk/ai-engineering-demos.git
   cd ai-engineering-demos
   ```

2. **Navigate to a project:**
   ```bash
   cd seq2seq-transformer-nmt
   ```

3. **Install project dependencies:**
   ```bash
   pip install -r requirements.txt  # if available
   # Or check the project README for specific installation instructions
   ```

4. **Run the project:**
   ```bash
   jupyter notebook  # For notebook-based projects
   # or
   python script_name.py  # For Python scripts
   ```

## 🎓 Skills Demonstrated

- **Deep Learning:** Neural network architectures, CNNs, transformers, attention mechanisms
- **Model Optimization:** Parameter-efficient fine-tuning (LoRA, PEFT), quantization, transfer learning
- **LLM Engineering:** Prompt engineering, instruction tuning, preference alignment (DPO/RLHF)
- **RAG Systems:** Vector databases, retrieval strategies, semantic search, document processing
- **Agentic AI:** Multi-agent coordination, tool integration, execution control, human-in-the-loop workflows
- **Production Practices:** Error handling, evaluation metrics, visualization, deployment-ready code

## 📊 Project Statistics

- **Total Projects:** 12
- **Lines of Code:** 10,000+
- **Jupyter Notebooks:** 8
- **Python Scripts:** 15+
- **Technologies:** 20+

## 🏆 Certifications

Projects completed as part of:
- **IBM AI Engineering Professional Certificate** (Coursera)
- **IBM Advanced RAG & Agentic AI Professional Certificate** (Coursera)

## 📖 Documentation

Each project includes:
- ✅ Comprehensive README with problem statement and architecture
- ✅ Code comments and documentation
- ✅ Installation and setup instructions
- ✅ Results and evaluation metrics
- ✅ Skills demonstrated section

## 🤝 Contributing

This is a personal portfolio repository showcasing certification work. While contributions are not expected, feedback and suggestions are welcome via issues.

## 📄 License

This project is licensed under the MIT License - see individual project directories for details.

## 🔗 Connect

- **GitHub:** [@deepdmk](https://github.com/deepdmk)
- **Portfolio Projects:** See pinned repositories for featured standalone projects

## 🙏 Acknowledgments

- IBM Skills Network for course materials and datasets
- Hugging Face for transformers and model hosting
- LangChain, CrewAI, and BeeAI communities
- Open-source ML/AI community

---

**Note:** Several projects have been extracted into standalone repositories with enhanced documentation. See the [Featured Projects](#-featured-projects) section for links to these dedicated repositories.
