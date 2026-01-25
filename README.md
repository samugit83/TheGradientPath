# Welcome to TheGradientPath 🚀

**Your comprehensive learning journey through modern Machine Learning, Deep Learning, and Artificial Intelligence - from fundamentals to production systems.**

This repository is a complete educational resource that bridges theory with practice, covering everything from foundational neural networks to cutting-edge AI agent systems and production ML deployment. Each project is designed to be hands-on, practical, and production-ready, with clear documentation, video tutorials, and runnable code.

---

## 📚 Table of Contents

- [AI Agents](#ai-agents)
- [Deep Learning with Keras](#deep-learning-with-keras)
- [PyTorch Projects](#pytorch-projects)
- [LLM Fine-Tuning](#llm-fine-tuning)
- [RAG Systems](#rag-systems-retrieval-augmented-generation)
- [Real-World Production Projects](#real-world-production-projects)
- [MCP Protocol](#mcp-protocol-from-scratch)
- [AI Security Engineering](#ai-security-engineering)
- [Getting Started](#getting-started)

---

## AI Agents

### **Comprehensive AI Agent Framework Benchmark**

**Location:** `AiAgents/AgentFrameworkBenchmark/`

A production-grade comparison of **7 major AI agent frameworks** implementing identical multi-agent systems to provide objective, real-world benchmarks.

#### Video Tutorial
**[Watch on YouTube](https://youtu.be/ZIflDkdvOSA)** - Complete framework comparison and implementation guide

#### Frameworks Compared
1. **LangChain/LangGraph** (🥇 284/360) - Best overall, maximum flexibility
2. **OpenAI Agents** (🥈 277/360) - Minimal code, native MCP support
3. **CrewAI** (🥉 249/360) - Rapid prototyping, simple delegation
4. **LlamaIndex** (227/360) - Balanced workflow architecture
5. **AutoGen** (195/360) - Enterprise async infrastructure
6. **Semantic Kernel** (178/360) - Microsoft ecosystem integration
7. **Vanilla Python** - Baseline with zero framework overhead

#### What's Benchmarked
- ✅ **Agent Orchestration** - Multi-agent coordination and routing
- ✅ **Tool Integration** - Custom tool creation and execution
- ✅ **State Management** - Complex state handling across agents
- ✅ **Memory Management** - Persistent conversation history
- ✅ **MCP Server Integration** - Model Context Protocol support
- ✅ **Production Features** - Guardrails, token tracking, structured output

#### System Architecture
Each implementation includes:
- **Orchestrator Agent** - Routes queries to specialized agents
- **Legal Expert Agent** - Handles law and legal topics
- **Operational Agent** - Manages programming and general queries
- **Tools** - Weather API, calculator, web search
- **MCP Integration** - Extended capabilities via Model Context Protocol

**[Full Documentation](AiAgents/AgentFrameworkBenchmark/README.md)**

---

## Deep Learning with Keras

### **Modern Neural Network Implementations**

**Location:** `Keras/`

Production-ready deep learning implementations using TensorFlow and Keras, from fundamentals to advanced architectures.

### 1. **Image Classification with MLP**
**Path:** `Keras/ImageClassificationWithMLP/`

- **Dataset:** MNIST handwritten digits
- **Architecture:** Multi-layer perceptron with Dropout and BatchNormalization
- **Features:** TensorBoard integration, Visualkeras architecture diagrams
- **Tools:** Dense layers, functional API, comprehensive logging

### 2. **Transformer-Based Text Generation**
**Path:** `Keras/transformers/text_generation/`

- **Task:** Natural language generation from scratch
- **Architecture:** Complete Transformer implementation
- **Components:** Multi-head self-attention, positional encoding, feed-forward networks
- **Features:** Custom training loop, text preprocessing, generation sampling

### 3. **Text Generation with KV Cache**
**Path:** `Keras/transformers/kv_cache_for text_gen/`

- **Optimization:** Key-Value cache for efficient inference
- **Performance:** Dramatically reduced computation during generation
- **Architecture:** Modified Transformer with caching mechanism
- **Use Case:** Production LLM inference optimization

### 4. **Time Series Forecasting with Transformers**
**Path:** `Keras/transformers/time_series_forecast/`

- **Task:** Stock price prediction using Transformers
- **Data:** Synthetic financial time series
- **Architecture:** Transformer adapted for sequential prediction
- **Features:** Temporal embeddings, MinMax scaling, visualization

**Key Learning Points:**
- Building Transformers from scratch in Keras
- Multi-head attention mechanisms
- Positional encoding strategies
- KV cache optimization techniques
- Adapting Transformers for different domains

---

## PyTorch Projects

### **CNN Image Classification**

**Location:** `Pytotch/CnnImageClassification/`

**[Full README](Pytotch/CnnImageClassification/README.md)**

#### Fashion-MNIST CNN Classifier
- **Dataset:** 70,000 images of 10 clothing categories
- **Architecture:** 2-layer CNN with BatchNorm
  - Conv2d(1→16) + BatchNorm + ReLU + MaxPool
  - Conv2d(16→32) + BatchNorm + ReLU + MaxPool
  - Fully Connected (512→10)
- **Performance:** ~85-90% validation accuracy
- **Features:** 
  - Automatic dataset download
  - GPU acceleration support
  - Model checkpointing
  - Training visualization
  - Real-time progress monitoring

**Key Learning Points:**
- Convolutional neural networks fundamentals
- Batch normalization for training stability
- PyTorch DataLoader and Dataset classes
- Model training and evaluation pipelines

---

## LLM Fine-Tuning

### **Advanced Parameter-Efficient Fine-Tuning Techniques**

**Location:** `LLMFineTuning/`

State-of-the-art techniques for efficiently fine-tuning large language models for specific tasks.

### 1. **All PEFT Techniques From Scratch**
**Path:** `LLMFineTuning/all_peft_tecniques_from_scratch/`

Complete implementation of Parameter-Efficient Fine-Tuning methods:
- **LoRA** (Low-Rank Adaptation) - Inject trainable low-rank matrices
- **Prefix Tuning** - Learn soft prompts prepended to inputs
- **Adapter Layers** - Small bottleneck layers inserted into models
- **IA³** (Infused Adapter by Inhibiting and Amplifying Inner Activations)

**Why PEFT?**
- Train only 0.1-1% of model parameters
- Reduce memory requirements by 90%
- Maintain performance close to full fine-tuning
- Enable multi-task learning with parameter isolation

### 2. **GRPO Reasoning with Unsloth**
**Path:** `LLMFineTuning/GRPO_REASONING_UNSLOTH/`

**Advanced reasoning capabilities through Gradient-based Reward Policy Optimization:**
- **Model:** Google Gemma 3 1B with 4-bit quantization
- **Technique:** GRPO (combines PPO benefits with gradient-based optimization)
- **Task:** Mathematical reasoning with structured outputs
- **Features:**
  - LoRA rank-32 adaptation
  - 4-bit quantization for memory efficiency
  - vLLM acceleration for fast inference
  - Structured reasoning format (`<reasoning>` and `<answer>` tags)

**Performance Gains:**
- Models learn to show reasoning steps
- Improved accuracy on complex problems
- Better interpretability of model decisions

### 3. **Supervised Fine-Tuning with Tool Choice**
**Path:** `LLMFineTuning/SFT_HF_TOOL_CHOICE/`

**Teaching models to intelligently select tools:**
- **Model:** HuggingFace SmolLM2-135M
- **Task:** Tool selection based on user queries
- **Dataset:** 10,000 synthetic examples with tool annotations
- **Technique:** Supervised Fine-Tuning with custom special tokens
- **Use Case:** Building function-calling capabilities in smaller models

**Real-World Application:**
- Enable LLMs to use external tools (calculators, APIs, databases)
- Reduce reliance on large models for specialized tasks
- Build cost-effective AI assistants

---

## RAG Systems (Retrieval-Augmented Generation)

### **Advanced RAG Architectures**

**Location:** `Rag/`

Production-ready Retrieval-Augmented Generation systems that enhance LLM responses with external knowledge.

### 1. **Dartboard RAG**
**Path:** `Rag/dartboard/`

**[Full README](Rag/dartboard/README.md)**

**Balanced Relevance and Diversity Retrieval**

Based on the paper: *"Better RAG using Relevant Information Gain"*

**Key Innovation:**
- **Problem:** Standard top-k retrieval returns redundant documents
- **Solution:** Optimize combined relevance-diversity score
- **Result:** Non-redundant, comprehensive context for LLMs

**Features:**
- Configurable relevance/diversity weights
- Production-ready modular design
- FAISS vector store integration
- Oversampling for better candidate selection

**Algorithm:**
```python
combined_score = diversity_weight * diversity + relevance_weight * relevance
```

**When to Use:**
- Dense knowledge bases with overlapping information
- Queries requiring diverse perspectives
- Avoiding echo chambers in retrieval

### 2. **Hybrid Multivector Knowledge Graph RAG**
**Path:** `Rag/hybrid_multivector_knowledge_graph_rag/`

**[Full README](Rag/hybrid_multivector_knowledge_graph_rag/README.md)**

**The Most Advanced RAG System - 11+ Graph Traversal Algorithms**

**Revolutionary Features:**
- **Knowledge Graph Engineering** with Neo4j
- **Multi-Vector Embeddings** for nuanced retrieval
- **11+ Graph Traversal Algorithms:**
  - K-hop Limited BFS
  - Depth-Limited DFS
  - A* Search with heuristics
  - Beam Search
  - Uniform Cost Search (UCS)
  - Context-to-Cypher query generation
  - LLM-powered intelligent filtering

**Architecture:**
1. **Vector Retrieval** - Initial similarity search
2. **Graph Traversal** - Navigate knowledge relationships
3. **Entity Extraction** - LLM-powered entity identification
4. **Dynamic Querying** - Context-aware Cypher generation
5. **Intelligent Ranking** - Multi-factor relevance scoring

**Why Knowledge Graphs?**
- Discover hidden connections across concepts
- Follow chains of reasoning
- Understand complex relationships
- Navigate multi-hop queries intelligently

**Use Cases:**
- Research and academic knowledge bases
- Legal document analysis
- Scientific literature review
- Complex domain expertise systems

### 3. **Vision RAG**
**Path:** `Rag/vision_rag/`

**Multimodal RAG for Documents with Images**

**Capabilities:**
- **PDF Processing** - Extract text and images from documents
- **Image Embeddings** - CLIP-based visual understanding
- **Unified Retrieval** - Search across text and images simultaneously
- **PostgreSQL + pgvector** - Scalable vector storage
- **Docker Deployment** - Production-ready containerization

**Architecture:**
- Text extraction and chunking
- Image extraction and captioning
- Dual embedding spaces (text + vision)
- Unified query interface
- Relevance-based ranking

**Use Cases:**
- Architectural design documents
- Scientific papers with diagrams
- Product catalogs
- Technical manuals
- Medical imaging reports

---

## Real-World Production Projects

### **ML Cyber Attack Prediction System**

**Location:** `RealWorldProjects/CyberAttackPrediction/`

**[Full README](RealWorldProjects/CyberAttackPrediction/README.md)** | **[Video Tutorial](https://youtu.be/3-mH1ynRf7U)**

**Enterprise-grade cloud-native ML system for real-time network threat detection.**

### Complete Production Stack

#### Infrastructure (AWS CloudFormation)
- **Application Load Balancer** - HTTPS/HTTP traffic distribution
- **Auto Scaling Groups** - Elastic capacity management
- **EC2 Instances** - Ubuntu 22.04 LTS compute
- **Target Groups** - Health-checked backend pools
- **Security Groups** - Network isolation and access control
- **IAM Roles** - Least-privilege security model

#### CI/CD Pipeline
- **AWS CodePipeline** - Automated deployment workflows
- **AWS CodeBuild** - Application compilation and testing
- **AWS CodeDeploy** - Zero-downtime deployments
- **S3 Artifact Storage** - Build artifact management
- **GitHub Integration** - Source control via CodeStar

### ML Architecture

**Multi-Stage Pipeline:**
1. **Data Preprocessing** - Mixed numerical/categorical feature handling
2. **AutoEncoder** - Learn normal traffic patterns, detect anomalies
3. **Feature Selection (ORC)** - Dynamic relevance-based feature selection
4. **SGD Classification** - Final attack prediction
5. **Incremental Learning** - Continuous model improvement

**Performance:**
- Real-time prediction (<1s response time)
- High accuracy on network attack detection
- Scalable to high traffic volumes

### System Components

#### 1. Monitor App (Next.js + Python)
- **Web Dashboard** - Real-time monitoring UI
- **Network Agent** - Scapy-based packet capture
- **Feature Extraction** - Flow-level statistics
- **RESTful API** - Health checks and metrics

#### 2. ML Service (Flask)
- **Prediction API** - RESTful inference endpoint
- **Model Management** - Load balancing and versioning
- **Batch Training** - Scheduled model updates
- **Metrics Tracking** - Performance monitoring

### One-Click Deployment

**CloudFormation Template Features:**
- Complete infrastructure as code
- Parameterized for easy customization
- Automatic DNS and SSL certificate setup
- Multi-AZ high availability
- Auto-scaling based on CPU utilization

**What Gets Deployed:**
```
┌─────────────────────────────────────────┐
│     Application Load Balancer          │
│         (HTTPS + HTTP)                  │
└──────────┬──────────────┬───────────────┘
           │              │
    ┌──────▼──────┐  ┌───▼──────────┐
    │ Monitor App │  │  ML Service  │
    │ Auto Scaling│  │  EC2 Instance│
    │   Group     │  │              │
    └─────────────┘  └──────────────┘
```

**Network Security:**
- TLS 1.3 encryption
- VPC isolation
- Security group restrictions
- IAM role-based access

---

### **RedAmon**

**Location:** `RealWorldProjects/RedAmon/`

---

## MCP Protocol From Scratch

### **Building Model Context Protocol Systems**

**Location:** `MCPFromScratch/`

**[Full README](MCPFromScratch/README.md)** | **[🎥 Video Tutorial](https://youtu.be/5KYZUtmQW_U)**

**Learn to build intelligent client-server AI systems from the ground up.**

### What You'll Build

#### Server Component (FastAPI)
- **Tools** - Calculator, database queries, text-to-SQL conversion
- **Prompts** - Reusable LLM interaction templates
- **Resources** - Configuration and data access
- **WebSocket Support** - Real-time bidirectional communication
- **Authentication** - API key validation and quota management

#### Client Component (Intelligent Agent)
- **Natural Language Understanding** - Parse user queries
- **Tool Discovery** - Automatically detect available capabilities
- **Dynamic Selection** - Choose appropriate tools based on context
- **Conversational Interface** - Friendly user interactions
- **OpenAI Integration** - LLM-powered intelligence

### Architecture Patterns

**Protocol Flow:**
```
Client                    Server
  │                         │
  ├─── Connect (WS) ────────►
  │                         │
  ├─── Initialize ──────────►
  │◄─── Capabilities ────────┤
  │                         │
  ├─── Call Tool ───────────►
  │◄─── Result ──────────────┤
  │                         │
  ├─── Get Prompt ──────────►
  │◄─── Template ────────────┤
```

### Key Concepts

- **Model Context Protocol** - Custom AI communication protocol
- **WebSocket Sessions** - Persistent connections for real-time interaction
- **Schema Validation** - Pydantic for robust data handling
- **Async Programming** - Modern Python concurrency with asyncio
- **API Design** - RESTful and WebSocket patterns

### Learning Path

1. **Understand the Protocol** - How clients and servers communicate
2. **Build the Server** - Implement tools, prompts, and resources
3. **Create the Client** - Build an intelligent agent
4. **Integration** - Connect components via WebSocket
5. **Enhancement** - Add custom tools and capabilities

**Perfect for:**
- Understanding AI agent architectures
- Building custom LLM-powered tools
- Learning modern async Python
- Designing extensible AI systems

---

## AI Security Engineering

### **STRIDE-Based Security Documentation**

**Location:** `AISecurityEngineering/`

**[Full README](AISecurityEngineering/README.md)**

**Comprehensive security engineering documentation using the STRIDE threat modeling framework, covering traditional web security and AI/LLM-specific attack vectors.**

### What is STRIDE?

STRIDE is a threat modeling framework developed by Microsoft:

| Category | Description |
|----------|-------------|
| **S - Spoofing** | Identity falsification attacks |
| **T - Tampering** | Unauthorized data modification |
| **R - Repudiation** | Denying performed actions |
| **I - Information Disclosure** | Exposing sensitive data |
| **D - Denial of Service** | Making systems unavailable |
| **E - Elevation of Privilege** | Gaining unauthorized access |

### Coverage (25 Documents)

#### Spoofing (12 documents)
- Brute Force Attacks, Credential Stuffing, Session Hijacking
- JWT Token Theft, Replay Attacks, Phishing
- Email/IP Spoofing, Clickjacking, SSRF
- Subdomain Takeover, **LLM Prompt Injection (LLM01:2025)**

#### Tampering (9 documents)
- SQL Injection, XSS, CSRF, Parameter Tampering
- Supply Chain Attacks
- **LLM Data Poisoning (LLM04:2025)**, **Improper Output Handling (LLM05:2025)**
- **Misinformation (LLM09:2025)**, **LLM Supply Chain (LLM03:2025)**

#### Information Disclosure (2 documents)
- **Sensitive Information Disclosure (LLM06:2025)**
- **Vector/Embedding Weaknesses (LLM08:2025)**

#### Denial of Service (1 document)
- **Unbounded Consumption (LLM10:2025)**

#### Elevation of Privilege (1 document)
- **Excessive Agency (LLM08:2025)**

### Document Structure

Each document includes:
- **Overview Mermaid Diagrams** - Visual attack flows
- **4 Real-World Attack Scenarios** - Step-by-step attack chains
- **Python PoC Scripts** - Proof-of-concept code for lab testing
- **AI/ML-Enhanced Attack Techniques** - How adversaries leverage AI
- **Mitigation Strategies** - Defensive countermeasures

**Who It's For:**
- Security researchers and penetration testers
- Developers learning secure coding practices
- AI/ML engineers understanding LLM vulnerabilities
- Security teams building threat models

---

## Getting Started

### Prerequisites

- **Python 3.8+** (3.10 recommended)
- **pip** or **conda** for package management
- **Git** for version control
- **OpenAI API Key** (for LLM-powered projects)
- **Docker** (optional, for containerized projects)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/samugit83/TheGradientPath.git
cd TheGradientPath

# Choose a project and navigate to it
cd <project_directory>

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Follow project-specific README for next steps
```

### Project-Specific Setup

Each project includes:
- **requirements.txt** - Python dependencies
- **README.md** - Detailed documentation
- **Video Tutorial** - Step-by-step guide (where available)
- **Jupyter Notebooks** - Interactive exploration

---

## Learning Philosophy

**TheGradientPath** is designed around these principles:

### 1. **Hands-On Learning**
Every concept is accompanied by runnable code. Learn by doing, not just reading.

### 2. **Production-Ready Code**
All implementations follow best practices and are designed for real-world use, not just tutorials.

### 3. **Comprehensive Documentation**
Each project includes detailed explanations, architecture diagrams, and video tutorials.

### 4. **Progressive Complexity**
Start with fundamentals (MLP, CNN) and progress to advanced systems (multi-agent RAG, production ML).

### 5. **Open Source & Accessible**
All code uses open-source libraries and can run on consumer hardware.

---

## Skill Progression Map

```
Beginner
├─ Keras MLP Image Classification
├─ PyTorch CNN Fundamentals
└─ Basic RAG (Dartboard)

Intermediate
├─ Transformer Text Generation
├─ LLM Fine-Tuning (SFT)
├─ Multi-vector RAG
└─ MCP Protocol

Advanced
├─ Knowledge Graph RAG
├─ GRPO Reasoning
├─ Vision RAG
└─ AI Agent Frameworks

Expert
├─ Production ML System
├─ Agent Framework Benchmark
└─ All PEFT Techniques
```

---

## Community & Contributing

### Getting Help
- **GitHub Issues** - Report bugs or request features
- **Discussions** - Ask questions and share insights
- **Video Comments** - Engage on YouTube tutorials

### Contributing
Contributions are welcome! Whether it's:
- Bug fixes
- Documentation improvements
- New features or projects
- Code quality enhancements

**To contribute:**
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

## Video Tutorials

Many projects include companion video tutorials:

- **[AI Agent Framework Benchmark](https://youtu.be/ZIflDkdvOSA)** - 7 frameworks compared
- **[ML Cyber Attack Prediction](https://youtu.be/3-mH1ynRf7U)** - Production ML system
- **[MCP From Scratch](https://youtu.be/5KYZUtmQW_U)** - Build intelligent client-server systems

**Subscribe for more!** Weekly deep dives into AI, ML, and production systems.

---

## About the Author

**Samuele Giampieri**  
*AI Engineer specializing in Knowledge Graphs, NLP, and AI-Driven Systems*

Passionate about bridging cutting-edge research with practical applications. Expertise spans:
- Knowledge graphs and graph neural networks
- Multi-agent systems and orchestration
- RAG architectures and information retrieval
- Production ML deployment and MLOps

### Connect
- **GitHub:** [github.com/samugit83](https://github.com/samugit83)
- **LinkedIn:** AI/ML discussions and networking
- **YouTube:** Weekly AI and ML tutorials
- **Email:** Consulting and collaboration inquiries

### Support This Project
- ⭐ **Star this repository** if you find it helpful
- 👍 **Like the videos** on YouTube
- 🔔 **Subscribe** for weekly content
- 💬 **Share** your projects and feedback
- 🤝 **Contribute** improvements

---

## License

This project is part of TheGradientPath educational initiative. Free to use for learning, research, and commercial applications.

---

## Acknowledgments

Special thanks to:
- The open-source community for incredible tools and libraries
- Researchers publishing papers and sharing knowledge
- Students and practitioners providing feedback
- Everyone contributing to democratizing AI education

---

**Built with ❤️ by Samuele Giampieri**

*Follow the gradient toward mastery, one project at a time.*

**Last Updated:** October 2025


