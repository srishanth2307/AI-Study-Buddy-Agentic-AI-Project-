# 📚 Personalized AI Study Buddy

## 🌟 Features

### 🤖 **Intelligent Q&A System**
- Ask any topic or question and receive clear, structured explanations
- Contextual follow-up questions with conversation memory
- Natural dialogue flow with short-term memory (Streamlit session state)

### 🧠 **Persistent Long-term Memory**
- ChromaDB vector store for semantic search
- Retrieve similar topics from previous study sessions
- Auto-saves all explanations for future reference

### 📝 **Interactive Quiz Generation**
- Automatically generates 3-5 multiple choice questions
- Interactive answer selection with radio buttons
- Real-time scoring and feedback
- Review correct answers with expandable sections

### 💾 **Notes Export**
- Save complete conversation history
- Timestamped entries for tracking progress
- Export to `notes/my_notes.txt` for offline access

### 🎨 **Modern UI**
- Dark theme interface
- Clean, intuitive design
- Progress indicators for long operations
- Error-free user experience

### 🔒 **Privacy & Security**
- 100% offline operation (after initial setup)
- No API keys or external services
- All data stored locally
- No data transmission to external servers

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+** (Python 3.10+ should work)
- **4GB+ RAM** (8GB recommended)
- **500MB+ disk space** for models
- **GGUF model file** (download separately)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd AGENTIC-AI-PROJECT
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download a GGUF model**
   
   Create a `models` directory and download a compatible model:
   ```bash
   mkdir models
   ```
   
   **Recommended models:**
   - [TinyLlama-1.1B-Chat](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) (~700MB) - Fast, lightweight
   - [Llama-2-7B-Chat](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF) (~4GB) - Better quality
   
   Place the `.gguf` file in the `./models/` directory.

4. **Run the application**
   
   **Option 1: Using Python directly**
   ```bash
   streamlit run app.py
   ```
   
   **Option 2: Using the startup script**
   - **Windows**: Double-click `start_app.bat`
   - **PowerShell**: Double-click `start_app.ps1` or run `.\start_app.ps1`
   
5. **Open in browser**
   
   The app will automatically open at `http://localhost:8501`
   
   If it doesn't open automatically, navigate to:
   ```
   http://localhost:8501
   ```

---

## 📖 Usage Guide

### Step 1: Load Your Model
1. Open the app in your browser
2. In the sidebar, click **"🔄 Load Model"**
3. The app will auto-detect your GGUF file in `./models/`
4. Wait for the success message

### Step 2: Start Learning
1. Enter any topic or question in the input field
   - Example: "What is machine learning?"
   - Example: "Explain quantum physics"
2. Click **"❓ Ask"** to get an explanation
3. Ask follow-up questions - the app remembers context!

### Step 3: Generate Quizzes
1. After asking a question, click **"📝 Generate Quiz"**
2. Select your answers using radio buttons
3. Click **"✅ Check Answers"** to see your score
4. Review correct answers in the expandable section

### Step 4: Save Your Notes
1. Click **"💾 Save Notes"** anytime
2. Your conversation history is saved to `./notes/my_notes.txt`
3. Access your notes offline anytime

### Step 5: Search Similar Topics
- Use the sidebar search to find related topics
- The vector store retrieves semantically similar content
- Great for reviewing previous study sessions

---

## 🏗️ Project Structure

```
AGENTIC-AI-PROJECT/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── start_app.bat         # Windows startup script
├── start_app.ps1         # PowerShell startup script
│
├── models/               # GGUF model files (place your .gguf here)
│   └── *.gguf
│
├── chroma_db/            # Vector store database (auto-created)
│   └── ...
│
└── notes/                # Saved study notes (auto-created)
    └── my_notes.txt      # Your exported notes
```

---

## 🛠️ Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **UI Framework** | Streamlit | Web interface and user interactions |
| **AI Orchestration** | LangChain | Workflow management and AI pipeline |
| **LLM Runtime** | llama-cpp-python | Local LLM inference (CPU-based) |
| **Vector Database** | ChromaDB | Persistent storage for embeddings |
| **Embeddings** | sentence-transformers | Semantic similarity and search |
| **Model Format** | GGUF | Efficient model quantization |

---

## ⚙️ Configuration

### Model Settings

You can adjust model parameters in `app.py` (line ~86):

```python
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.7,        # Creativity (0.0-1.0)
    max_tokens=150,         # Response length
    n_ctx=512,              # Context window size
    n_threads=4,            # CPU threads (adjust based on your CPU)
    # ... more settings
)
```

### Performance Optimization

- **Faster responses**: Reduce `max_tokens` and `n_ctx`
- **Better quality**: Use larger models or increase `max_tokens`
- **Memory usage**: Adjust `n_threads` based on available RAM

---

## 🐛 Troubleshooting

### Model Not Loading
- ✅ Ensure the `.gguf` file is in `./models/` directory
- ✅ Check that the model file is not corrupted
- ✅ Try a smaller model first (TinyLlama works well)

### Embeddings Error (Meta Tensor)
- ✅ This is handled automatically by the app
- ✅ The app will work without embeddings (vector store disabled)
- ✅ Try clicking "🔄 Reload Embeddings" in the sidebar

### App Not Starting
- ✅ Check Python version: `python --version` (should be 3.10+)
- ✅ Install dependencies: `pip install -r requirements.txt`
- ✅ Check if port 8501 is available
- ✅ Use the startup scripts (`start_app.bat` or `start_app.ps1`)

### Slow Responses
- ✅ Use a smaller/quantized model (Q4_K_M or smaller)
- ✅ Reduce `max_tokens` in model settings
- ✅ Close other applications to free up CPU/RAM
- ✅ First response is always slower (model loading)

### Quiz Not Generating
- ✅ Ensure you've asked a question first
- ✅ Check that the model is loaded
- ✅ Try asking a simpler question
- ✅ The app has fallback questions if generation fails

---

## 📊 System Architecture

```
┌─────────────────────────────────────────┐
│      Streamlit UI (User Interface)      │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│      LangChain (Orchestration)          │
└───────┬───────────────────┬─────────────┘
        │                   │
┌───────▼───────┐   ┌───────▼────────────┐
│  Local LLM    │   │  ChromaDB Vector   │
│ (llama-cpp)   │   │  Store             │
│               │   │  - Embeddings      │
│ - TinyLlama   │   │  - Similarity      │
│ - GGUF Format │   │    Search          │
└───────────────┘   └────────────────────┘
```

---

## 🔄 Workflow

1. **User Input** → Streamlit captures question
2. **Prompt Engineering** → LangChain formats prompt with context
3. **LLM Inference** → llama-cpp-python generates response
4. **Response Display** → Streamlit shows explanation
5. **Vector Storage** → ChromaDB saves for future retrieval
6. **Embedding Generation** → sentence-transformers creates embeddings

---

## 🎯 Key Features Explained

### Conversation Memory
- **Short-term**: Maintained via Streamlit `session_state`
- **Long-term**: Stored in ChromaDB vector store
- Enables contextual follow-up questions

### Vector Store
- Uses semantic embeddings (all-MiniLM-L6-v2)
- Enables similarity search across all studied topics
- Persists between sessions

### Quiz Generation
- LLM generates JSON-formatted questions
- Fallback questions if generation fails
- Interactive UI with real-time feedback

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Ideas
- [ ] Add support for GPU acceleration
- [ ] Implement different quiz types (True/False, Fill-in-blank)
- [ ] Add study analytics and progress tracking
- [ ] Support for multiple languages
- [ ] Export notes in different formats (PDF, Markdown)
- [ ] Add code syntax highlighting for programming topics

---

## 🙏 Acknowledgments

- **TinyLlama** - Lightweight LLM for local inference
- **LangChain** - AI application framework
- **ChromaDB** - Vector database for embeddings
- **sentence-transformers** - Semantic embeddings
- **Streamlit** - Rapid web app development

---

## ⚡ Performance Tips

- **First run**: Embeddings model downloads automatically (~80MB)
- **Model loading**: Takes 10-30 seconds on first load
- **Response time**: 5-15 seconds (depends on model size and CPU)
- **Memory usage**: 2-4GB RAM (varies with model size)

---

## 🔮 Future Enhancements

- [ ] Multiple model support with easy switching
- [ ] Advanced quiz types and difficulty levels
- [ ] Study session statistics and analytics
- [ ] Topic tagging and organization
- [ ] Calendar integration for study reminders
- [ ] Collaborative features for study groups
- [ ] Mobile app version
- [ ] Cloud sync (optional, privacy-preserving)

---



