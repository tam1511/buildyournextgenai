# Tài Liệu Học AI 2026 — Theo Từng Tầng

> Tổng hợp nguồn học chất lượng cho từng tầng trong **AI Career Roadmap 2026**.  
> Ưu tiên nguồn **miễn phí**, chất lượng học thuật cao, kiểm chứng thực tế.

---

## Mục Lục

- [Core Path Tóm Tắt](#-core-path-tóm-tắt)
- [Tầng 1 — Foundation](#tầng-1--foundation)
- [Tầng 2 — Python Ecosystem](#tầng-2--python-ecosystem)
- [Tầng 3 — Machine Learning](#tầng-3--machine-learning)
- [Tầng 4 — Deep Learning & LLM Engineering](#tầng-4--deep-learning--llm-engineering)
- [Tầng 5 — Production & MLOps](#tầng-5--production--mlops)
- [Bonus — Papers Nền Tảng](#bonus--papers-nền-tảng)
- [Cộng Đồng & Nguồn Cập Nhật](#cộng-đồng--nguồn-cập-nhật)

---

## Core Path Tóm Tắt

> Nếu bạn chỉ muốn con đường tinh gọn nhất — học đúng thứ này, theo đúng thứ tự này.

| Thứ tự | Tài liệu | Tầng | Ghi chú |
|--------|----------|------|---------|
| 1 | [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | 1 | Xem trước |
| 2 | [CS50P — Python (Harvard)](https://cs50.harvard.edu/python/2022/) | 1 | Nền tảng lập trình cho người mới |
| 3 | [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) | 2 | NumPy, Pandas, Matplotlib — đọc free online |
| 4 | [ML Specialization — Andrew Ng](https://www.coursera.org/specializations/machine-learning-introduction) | 3 | Audit miễn phí. |
| 5 | [Hands-On Machine Learning — Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) | 3 | Sách thực chiến nhất, bản cũ free trên GitHub |
| 6 | [Neural Networks: Zero to Hero — Karpathy](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | 4 | Build GPT từ đầu |
| 7 | [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1) | 4 | Transformers → fine-tuning, miễn phí |
| 8 | [Prompt Engineering Guide (DAIR.AI)](https://www.promptingguide.ai/) | 4 | Tổng hợp đầy đủ nhất về prompting |
| 9 | [Building Effective Agents — Anthropic](https://www.anthropic.com/research/building-effective-agents) | 4 | System design cho Agent — đọc kỹ |
| 10 | [Made With ML](https://madewithml.com/) | 5 | MLOps end-to-end tốt nhất, miễn phí |

**Sau Core Path:** Build 3–5 project production-ready. Portfolio quan trọng hơn certificate.

---

## Tầng 1 — Foundation

> ~4–8 tuần &nbsp;|&nbsp; Mục tiêu: Có đủ toán cơ bản, viết được Python, đọc được tài liệu tiếng Anh

---

### Toán Cơ Bản

| Tài liệu | Loại | Ghi chú |
|----------|------|---------|
| [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Video | Trực quan nhất để hiểu vector & ma trận. **Bắt đầu từ đây.** |
| [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) | Video | Hiểu đạo hàm và gradient bằng hình ảnh |
| [StatQuest with Josh Starmer](https://www.youtube.com/@statquest) | Video | Xác suất & thống kê giải thích cực rõ, xem theo từng chủ đề |
| [Khan Academy — Statistics & Probability](https://www.khanacademy.org/math/statistics-probability) | Web | Học từ cơ bản, miễn phí, có bài tập |
| [Mathematics for Machine Learning (Coursera)](https://www.coursera.org/specializations/mathematics-machine-learning) | Course | Deep Dive — Imperial College London, audit miễn phí |
| [The Matrix Calculus You Need for Deep Learning](https://arxiv.org/abs/1802.01528) | Paper | Deep Dive — PDF ngắn gọn, đủ dùng cho DL |

> Với mục tiêu **xây ứng dụng**, chỉ cần hiểu *ý nghĩa* và *khi nào dùng* — không cần tính tay. Thư viện đã làm điều đó thay bạn.

---

### Lập Trình Python

| Tài liệu | Loại | Ghi chú |
|----------|------|---------|
| [CS50P — Introduction to Programming with Python (Harvard)](https://cs50.harvard.edu/python/2022/) | Course | Tốt nhất cho người mới. Miễn phí. |
| [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/) | Sách | Đọc online miễn phí, rất thực tế |
| [Real Python](https://realpython.com/) | Web | Bài viết theo từng chủ đề cụ thể, chất lượng cao |
| [Python Official Tutorial](https://docs.python.org/3/tutorial/) | Docs | Nguồn chuẩn nhất, dùng để tra cứu |
| [Python for Everybody (Coursera)](https://www.coursera.org/specializations/python) | Course | Deep Dive — University of Michigan |

**Công cụ khuyến nghị:**

| Giai đoạn | Công cụ | Lý do |
|-----------|---------|-------|
| Mới bắt đầu | [Google Colab](https://colab.research.google.com/) | Chạy code ngay trên trình duyệt, không cài gì |
| Đã quen | [Jupyter Notebook](https://jupyter.org/) | Cài local, debug từng cell dễ dàng |
| Dự án thực tế | [VS Code](https://code.visualstudio.com/) | IDE chuyên nghiệp, nhiều extension hỗ trợ |

> Đừng vội dùng thư viện AI khi chưa hiểu Python thuần. Nền tảng này quyết định khả năng debug về sau.

---

### Tiếng Anh Đọc Hiểu

> Thay vì Google bằng tiếng Việt, hãy bắt đầu search bằng tiếng Anh cho **mọi vấn đề kỹ thuật**. Đây không phải kỹ năng phụ — đây là lợi thế cạnh tranh thực sự.

---

## Tầng 2 — Python Ecosystem

> ~2–4 tuần &nbsp;|&nbsp; Mục tiêu: Thành thạo NumPy, Pandas, biết visualize dữ liệu

### NumPy, Pandas & Visualization

| Tài liệu | Loại | Ghi chú |
|----------|------|---------|
| [Python Data Science Handbook (Jake VanderPlas)](https://jakevdp.github.io/PythonDataScienceHandbook/) | Sách | Miễn phí online. Bao phủ NumPy, Pandas, Matplotlib, Scikit-Learn. |
| [Kaggle — Pandas Course](https://www.kaggle.com/learn/pandas) | Course | Miễn phí, có notebook thực hành ngay, ~4 giờ |
| [Kaggle — Data Visualization Course](https://www.kaggle.com/learn/data-visualization) | Course | Seaborn cơ bản đến nâng cao |
| [NumPy Official Quickstart](https://numpy.org/doc/stable/user/quickstart.html) | Docs | Nắm cơ bản nhanh |
| [Pandas Official Getting Started](https://pandas.pydata.org/docs/getting_started/intro_tutorials/) | Docs | Tutorial chính thức, đầy đủ nhất |
| [Plotly Documentation](https://plotly.com/python/) | Docs | Deep Dive — Biểu đồ interactive, dùng nhiều trong production |

> Tải một dataset trên [Kaggle](https://www.kaggle.com/datasets) (Titanic, house prices), tự khám phá bằng Pandas và vẽ biểu đồ **trước khi xem solution**. Cách học hiệu quả nhất ở tầng này.

---

## Tầng 3 — Machine Learning

> ~6–10 tuần &nbsp;|&nbsp; Mục tiêu: Hiểu bản chất thuật toán, build và evaluate model end-to-end  
> **Đây là tầng quan trọng nhất. Đừng bỏ qua dù bạn chỉ muốn làm GenAI.**

---

### Khóa Học Chính

| Tài liệu | Loại | Ghi chú |
|----------|------|---------|
| [Machine Learning Specialization — Andrew Ng (Coursera)](https://www.coursera.org/specializations/machine-learning-introduction) | Course | Chuẩn nhất để hiểu nền tảng. Audit miễn phí. **Bắt buộc.** |
| [Hands-On Machine Learning — Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) | Sách | Sách thực chiến nhất hiện nay. Bản cũ free trên GitHub. |
| [StatQuest ML Playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) | Video | Giải thích từng thuật toán cực kỳ trực quan |
| [fast.ai — Practical Machine Learning](https://course.fast.ai/) | Course | Deep Dive — Học top-down, code trước rồi lý thuyết sau |

---

### Thư Viện & Công Cụ

| Tài liệu | Loại | Ghi chú |
|----------|------|---------|
| [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html) | Docs | Đọc kỹ phần pipeline và model evaluation |
| [Kaggle Learn — Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning) | Course | Missing values, pipelines, cross-validation |
| [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/) | Docs | Model mạnh nhất cho tabular data |
| [LightGBM Documentation](https://lightgbm.readthedocs.io/en/stable/) | Docs | Deep Dive — Nhanh hơn XGBoost, dùng nhiều trong production |

---

### Đánh Giá Mô Hình

| Tài liệu | Loại | Ghi chú |
|----------|------|---------|
| [Scikit-Learn — Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) | Docs | Đầy đủ nhất về metrics, cross-validation |
| [A Visual Introduction to Machine Learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/) | Web | Animation giải thích train/test split và overfitting |

---

### Dự Án Thực Hành Gợi Ý

```
Mức 1 — Titanic Survival Prediction       classification cơ bản
Mức 2 — House Price Prediction             regression + feature engineering
Mức 3 — Customer Churn Prediction          imbalanced data, business context
Mức 4 — Credit Card Fraud Detection        production mindset, cost-sensitive metrics
```

> Mục tiêu không phải là accuracy cao — mà là **đi hết quy trình**: EDA → feature engineering → train → evaluate → cải thiện. Làm ít nhất 2–3 lần.

---

## Tầng 4 — Deep Learning & LLM Engineering

> ~8–16 tuần &nbsp;|&nbsp; Mục tiêu: Hiểu kiến trúc DL, build RAG system, thiết kế AI Agent

---

### Deep Learning — Nền Tảng

| Tài liệu | Loại | Ghi chú |
|----------|------|---------|
| [Andrej Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | Video | Build GPT từ đầu bằng Python. Khó nhưng xứng đáng. **Rất khuyến nghị.** |
| [Deep Learning Specialization — Andrew Ng (Coursera)](https://www.coursera.org/specializations/deep-learning) | Course | Chuẩn nhất để hiểu neural network. Audit miễn phí. |
| [fast.ai — Practical Deep Learning](https://course.fast.ai/) | Course | Tiếp cận từ ứng dụng thực tế |
| [PyTorch Official Tutorials](https://pytorch.org/tutorials/) | Docs | Framework DL phổ biến nhất hiện nay |
| [Deep Learning Book — Goodfellow et al.](https://www.deeplearningbook.org/) | Sách | Deep Dive — Miễn phí online, dùng làm reference |

---

### LLM Engineering — Prompt & RAG

| Tài liệu | Loại | Ghi chú |
|----------|------|---------|
| [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1) | Course | Miễn phí, bao phủ transformers → fine-tuning. **Bắt buộc.** |
| [Prompt Engineering Guide (DAIR.AI)](https://www.promptingguide.ai/) | Web | Tổng hợp đầy đủ nhất về prompting techniques |
| [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) | Docs | Best practices cho Claude, áp dụng được cho mọi LLM |
| [OpenAI Cookbook](https://cookbook.openai.com/) | Web | Ví dụ thực tế: RAG patterns, evaluations, production |
| [LangChain Documentation](https://python.langchain.com/docs/introduction/) | Docs | Framework phổ biến nhất để build LLM apps |
| [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/) | Docs | Mạnh hơn LangChain cho RAG use cases |
| [Chip Huyen — LLM Evaluation](https://huyenchip.com/2024/01/16/llm-evaluation.html) | Blog | Bài viết đầy đủ nhất về cách đánh giá LLM |

---

### Vector Database & Embeddings

| Tài liệu | Loại | Ghi chú |
|----------|------|---------|
| [Pinecone — Learn Vector Database](https://www.pinecone.io/learn/) | Web | Bài viết chất lượng cao về embeddings, vector search |
| [ChromaDB Documentation](https://docs.trychroma.com/) | Docs | Vector DB phổ biến nhất cho local/small projects |
| [Weaviate Academy](https://weaviate.io/developers/academy) | Course | Miễn phí, học vector search từ cơ bản đến nâng cao |
| [Qdrant Documentation](https://qdrant.tech/documentation/) | Docs | Deep Dive — Production-grade vector DB |

---

### LLM Evaluation

| Tài liệu | Loại | Ghi chú |
|----------|------|---------|
| [RAGAS Documentation](https://docs.ragas.io/en/stable/) | Docs | Framework đánh giá RAG system tự động |
| [LangSmith](https://www.langchain.com/langsmith) | Tool | Tracing, testing và evaluation cho LLM apps |
| [Evals (OpenAI)](https://github.com/openai/evals) | GitHub | Framework eval từ OpenAI, nhiều ví dụ thực tế |

---

### AI Agent

| Tài liệu | Loại | Ghi chú |
|----------|------|---------|
| [Building Effective Agents (Anthropic)](https://www.anthropic.com/research/building-effective-agents) | Blog | System design cho agent đúng cách. **Đọc kỹ.** |
| [OpenAI — Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) | Docs | Nền tảng để hiểu tool use trong agents |
| [Anthropic — Tool Use Guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) | Docs | Tool use với Claude |
| [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) | Docs | Deep Dive — Build stateful agents |
| [AutoGen Documentation (Microsoft)](https://microsoft.github.io/autogen/) | Docs | Deep Dive — Multi-agent framework |

> Framework (LangChain, AutoGen...) chỉ là implementation detail. Quan trọng hơn là hiểu **system design phía sau**: luồng dữ liệu, điểm lỗi, cơ chế fallback.

---

### Fine-tuning & Open Source Models

| Tài liệu | Loại | Ghi chú |
|----------|------|---------|
| [Hugging Face — Fine-tuning Guide](https://huggingface.co/docs/transformers/training) | Docs | Chuẩn nhất cho fine-tuning với Transformers |
| [Unsloth](https://github.com/unslothai/unsloth) | GitHub | Fine-tuning nhanh gấp 2x, ít VRAM hơn 70% |
| [Ollama](https://ollama.com/) | Tool | Chạy open-source LLM local cực dễ |
| [LM Studio](https://lmstudio.ai/) | Tool | GUI để chạy và thử nghiệm local models |
| [Open LLM Leaderboard (Hugging Face)](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) | Web | So sánh hiệu suất các open-source models |

---

## Tầng 5 — Production & MLOps

> ~8–12 tuần &nbsp;|&nbsp; Mục tiêu: Deploy AI system thực tế, monitoring, cost optimization  
> **Đây là nơi phân biệt demo với sản phẩm thật.**

---

### MLOps & System Design

| Tài liệu | Loại | Ghi chú |
|----------|------|---------|
| [Made With ML (Goku Mohandas)](https://madewithml.com/) | Web | Tốt nhất cho MLOps end-to-end. Miễn phí.|
| [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/2022/) | Course | Từ training đến deployment, rất thực tế |
| [MLOps Zoomcamp (DataTalks.Club)](https://github.com/DataTalks-Club/mlops-zoomcamp) | Course | Miễn phí, hands-on với MLflow, Prefect, deployment |
| [CS329S — ML Systems Design (Stanford)](https://stanford-cs329s.github.io/) | Course | Deep Dive — Slides và materials miễn phí |

---

### Công Cụ MLOps

| Tài liệu | Loại | Ghi chú |
|----------|------|---------|
| [Docker — Get Started](https://docs.docker.com/get-started/) | Docs | Containerize AI apps. Bắt buộc cho production. |
| [FastAPI Documentation](https://fastapi.tiangolo.com/) | Docs | Build AI API nhanh nhất bằng Python |
| [MLflow Documentation](https://mlflow.org/docs/latest/index.html) | Docs | Experiment tracking, model registry, deployment |
| [Weights & Biases (W&B)](https://docs.wandb.ai/) | Docs | Experiment tracking mạnh nhất, free tier đủ dùng |
| [DVC (Data Version Control)](https://dvc.org/doc) | Docs | Deep Dive — Version control cho data và models |

---

### LLM Production

| Tài liệu | Loại | Ghi chú |
|----------|------|---------|
| [LangFuse](https://langfuse.com/docs) | Docs | Open-source LLM observability, self-hostable |
| [LiteLLM Documentation](https://docs.litellm.ai/) | Docs | Unified interface cho nhiều LLM providers, cost tracking |
| [Guardrails AI](https://www.guardrailsai.com/docs) | Docs | Kiểm soát và validate output của LLM |
| [Prometheus + Grafana](https://prometheus.io/docs/introduction/overview/) | Docs | Deep Dive — Monitoring stack phổ biến nhất |
| [Portkey Documentation](https://portkey.ai/docs) | Docs | Deep Dive — LLM gateway, fallback, monitoring |

---

### Cloud & Deployment

| Tài liệu | Loại | Ghi chú |
|----------|------|---------|
| [Hugging Face Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) | Docs | Deploy model nhanh nhất, không cần setup infra |
| [Railway](https://railway.app/) / [Render](https://render.com/) | Tool | Deploy FastAPI app, free tier có sẵn |
| [Google Cloud — Vertex AI](https://cloud.google.com/vertex-ai/docs) | Docs | Deep Dive — Managed ML platform từ Google |
| [AWS SageMaker](https://aws.amazon.com/sagemaker/) | Web | Deep Dive — ML deployment ở quy mô enterprise |

---

## Bonus — Papers Nền Tảng

> Không cần đọc hết — nhưng nên biết những papers này tồn tại và hiểu ý chính.

| Paper | Năm | Tại sao quan trọng |
|-------|-----|-------------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Transformer — kiến trúc của hầu hết LLM hiện đại |
| [BERT](https://arxiv.org/abs/1810.04805) | 2018 | Nền tảng của transfer learning trong NLP |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | 2020 | Giải thích tại sao scale hoạt động |
| [GPT-3 — Few-Shot Learners](https://arxiv.org/abs/2005.14165) | 2020 | Chứng minh sức mạnh của scale |
| [RAG — Lewis et al.](https://arxiv.org/abs/2005.11401) | 2020 | Nền tảng lý thuyết của RAG |
| [InstructGPT (RLHF)](https://arxiv.org/abs/2203.02155) | 2022 | Cách align LLM với human preference |
| [LoRA](https://arxiv.org/abs/2106.09685) | 2021 | Kỹ thuật fine-tuning hiệu quả nhất hiện nay |
| [ReAct](https://arxiv.org/abs/2210.03629) | 2022 | Nền tảng lý thuyết của AI Agent |

---

## Cộng Đồng & Nguồn Cập Nhật

---

### Cộng Đồng

| Nguồn | Mô tả |
|-------|-------|
| [Hugging Face Discord](https://discord.com/invite/hugging-face-879548962464493619) | Community lớn nhất về open-source AI |
| [r/MachineLearning](https://www.reddit.com/r/MachineLearning/) | Thảo luận research và industry news |
| [Kaggle Discussion](https://www.kaggle.com/discussions) | Học từ solution của người khác |
| [LangChain Discord](https://discord.com/invite/langchain-1076006462262077491) | Support và chia sẻ projects |

---

### Thực Hành & Projects

| Nguồn | Mô tả |
|-------|-------|
| [Kaggle Competitions](https://www.kaggle.com/competitions) | Thi đấu thực tế, portfolio builder |
| [Papers With Code](https://paperswithcode.com/) | SOTA models với code implementation |
| [awesome-llm](https://github.com/Hannibal046/Awesome-LLM) | Tổng hợp resources về LLM |
| [awesome-mlops](https://github.com/visenger/awesome-mlops) | Tổng hợp resources về MLOps |

---

## Ghi Chú

```
Không cần học hết — chọn 1-2 nguồn mỗi tầng và đi sâu
Làm project thực tế quan trọng hơn đọc nhiều tài liệu
Deep Dive = học sau khi nắm vững Core Path
File này sẽ được cập nhật theo thời gian
```

> Bạn Có tài liệu hay muốn đóng góp? Mở **Pull Request** hoặc tạo **Issue**.  
> Theo dõi series video: **[[Link YouTube Channel](https://youtu.be/KUf8jOgFV0E)]**

---

**Nếu bạn muốn học theo lộ trình có cấu trúc có thể tham gia khoá học của mình trên Udemy [https://www.udemy.com/course/ai-engineer-thuc-chien-ml-rag-llms-agents-production/?couponCode=2262DDF3A4F3F4060157]**

---

*Cập nhật: 2026 · AI Career Roadmap Series*
