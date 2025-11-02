"""
Personalized AI Study Buddy - A complete offline AI learning assistant
Built with Streamlit, LangChain, llama-cpp-python, and Chroma
"""

import streamlit as st
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime

# LangChain imports
try:
    # Try newer LangChain imports first (for langchain >= 0.1.0)
    from langchain_community.llms import LlamaCpp
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    # Fallback to older imports (for langchain < 0.1.0)
    from langchain.llms import LlamaCpp
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings

# LangChain core imports (used for document schema)
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain_core.documents import Document

# Note: sentence-transformers is used internally by HuggingFaceEmbeddings

# Initialize session state for conversation memory
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "current_topic" not in st.session_state:
    st.session_state.current_topic = ""

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ["./chroma_db", "./notes", "./models"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        st.info(f"✓ Directory ready: {directory}")

def load_local_llm(model_path: str = None):
    """
    Load local LLM using llama-cpp-python
    If model_path is None, tries to find a .gguf or .bin file in ./models directory
    """
    if st.session_state.llm is not None:
        return st.session_state.llm
    
    try:
        # Try to find a model file in the models directory
        if model_path is None:
            model_dir = Path("./models")
            model_files = list(model_dir.glob("*.gguf")) + list(model_dir.glob("*.bin"))
            if model_files:
                model_path = str(model_files[0])
            else:
                # Use a small default model name (user needs to download)
                st.error("⚠️ No model file found in ./models directory.")
                st.info("📥 Please download a GGUF model (e.g., from Hugging Face) and place it in ./models/")
                st.info("Example: Download a small model like 'TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf'")
                return None
        
        # Initialize LlamaCpp with the model
        # Optimized settings for faster inference
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.7,
            max_tokens=150,  # Further reduced for faster responses
            n_ctx=512,  # Smaller context window for speed
            n_threads=4,  # Use 4 CPU threads
            n_batch=128,  # Smaller batch for faster processing
            verbose=False,
            n_predict=150,  # Limit prediction tokens
            repeat_penalty=1.1,  # Prevent repetition
            streaming=False,  # Disable streaming for now
            use_mmap=True,  # Use memory mapping for faster loading
            use_mlock=False  # Don't lock memory (faster)
        )
        
        st.session_state.llm = llm
        return llm
    except Exception as e:
        st.error(f"❌ Error loading LLM: {str(e)}")
        st.info("💡 Make sure you have a compatible GGUF model file in ./models/")
        return None

def load_embeddings():
    """Load sentence-transformers model for generating embeddings"""
    if st.session_state.embeddings is not None:
        return st.session_state.embeddings
    
    try:
        import os
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        # Set environment variables to avoid meta tensor issues
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        
        st.info("📥 Loading embeddings model (first time may take a moment)...")
        
        # Fix meta tensor issue by loading model components directly
        # This bypasses the lazy loading that causes meta tensor problems
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.models import Transformer, Pooling
        
        model_name = 'all-MiniLM-L6-v2'
        full_model_name = f'sentence-transformers/{model_name}'
        
        try:
            # Method 1: Try standard loading first
            model = SentenceTransformer(model_name, device='cpu')
        except Exception as e1:
            if "meta tensor" in str(e1).lower() or "to_empty" in str(e1).lower():
                # Meta tensor issue - load components manually
                st.warning("⚠️ Using manual model loading to fix meta tensor issue...")
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(full_model_name)
                
                # Load base model WITHOUT lazy loading (critical fix)
                base_model = AutoModel.from_pretrained(
                    full_model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,  # Critical: Disable lazy loading
                    device_map=None  # Don't use device_map (can cause meta tensors)
                )
                
                # Explicitly move to CPU immediately (before eval)
                base_model = base_model.to('cpu')
                base_model.eval()
                
                # Verify it's not meta tensor
                for param in base_model.parameters():
                    if hasattr(param, 'device') and 'meta' in str(param.device):
                        raise Exception("Still has meta tensors after loading")
                
                # Build SentenceTransformer manually
                word_embedding_model = Transformer(
                    tokenizer=tokenizer,
                    model=base_model,
                    max_seq_length=256
                )
                
                pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
                
                # Construct the SentenceTransformer
                model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
                model.eval()
            else:
                # Different error - try simple loading
                model = SentenceTransformer(model_name, device='cpu')
        
        # Final verification: Test the model works
        try:
            test_output = model.encode("test", convert_to_numpy=True)
            if test_output is None or len(test_output) == 0:
                raise Exception("Model encoding test failed")
        except Exception as test_err:
            st.error(f"Model loaded but encoding failed: {str(test_err)}")
            raise
        
        # Create embedding wrapper class compatible with LangChain
        class LangChainEmbeddingWrapper:
            """Wrapper to make SentenceTransformer compatible with LangChain"""
            def __init__(self, model):
                self.model = model
            
            def embed_documents(self, texts):
                """Embed multiple documents"""
                if not texts:
                    return []
                # Handle both list and single string
                if isinstance(texts, str):
                    texts = [texts]
                embeddings = self.model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                # Convert to list of lists
                if len(embeddings.shape) == 1:
                    return [embeddings.tolist()]
                return embeddings.tolist()
            
            def embed_query(self, text):
                """Embed a single query"""
                if not text:
                    return []
                embedding = self.model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                return embedding.tolist()
        
        # Create wrapper instance
        wrapper = LangChainEmbeddingWrapper(model)
        
        # Create object compatible with HuggingFaceEmbeddings interface
        # This will work with Chroma
        class CompatibleEmbeddings:
            def __init__(self, wrapper):
                self.wrapper = wrapper
                # Add these attributes that Chroma might check
                self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
            def embed_documents(self, texts):
                return self.wrapper.embed_documents(texts)
            
            def embed_query(self, text):
                return self.wrapper.embed_query(text)
        
        embeddings = CompatibleEmbeddings(wrapper)
        
        st.session_state.embeddings = embeddings
        st.success("✅ Embeddings loaded successfully!")
        return embeddings
            
    except Exception as e:
        st.error(f"❌ Error loading embeddings: {str(e)}")
        st.warning("⚠️ Vector store features (long-term memory) will be disabled.")
        st.info("💡 Core features (Q&A, Quiz, Notes) will still work perfectly!")
        return None

def initialize_vector_store():
    """Initialize or load Chroma vector store for persistent memory"""
    if st.session_state.vector_store is not None:
        return st.session_state.vector_store
    
    try:
        embeddings = load_embeddings()
        if embeddings is None:
            return None
        
        # Initialize Chroma vector store with persistent storage
        # Use collection_name to ensure consistency
        vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
            collection_name="study_buddy_memory"
        )
        
        # Test the embeddings work by doing a simple embed
        try:
            test_embed = embeddings.embed_query("test")
            if not test_embed or len(test_embed) == 0:
                st.warning("⚠️ Embeddings test failed - vector store may not work correctly.")
        except Exception as test_error:
            st.warning(f"⚠️ Embeddings test error: {str(test_error)}")
        
        st.session_state.vector_store = vector_store
        return vector_store
    except Exception as e:
        # Show warning but don't block the app
        st.warning(f"⚠️ Vector store initialization failed: {str(e)}")
        st.info("💡 App will work without vector store. Try reloading embeddings if needed.")
        return None

# ============================================================================
# CORE AI FUNCTIONS
# ============================================================================

def generate_explanation(topic: str, question: str = None) -> str:
    """
    Generate an explanation for a topic or question using the local LLM
    """
    llm = st.session_state.llm
    if llm is None:
        return "❌ LLM not loaded. Please ensure a model file is available."
    
    # Build a concise prompt for faster generation
    if question:
        prompt = f"Q: {question}\nA:"
    else:
        prompt = f"Explain: {topic}\n\n"
    
    try:
        # Minimal context - only last exchange if available
        if st.session_state.conversation_history:
            last = st.session_state.conversation_history[-1]
            if len(last.get('user', '')) < 50:  # Only add if short
                prompt = f"Context: {last.get('user', '')}\n\n{prompt}"
        
        # Use invoke() for newer LangChain versions, fallback to direct call for older versions
        try:
            response = llm.invoke(prompt)
        except AttributeError:
            # Older version - try direct call
            response = llm(prompt)
        
        # Handle response if it's a string or needs extraction
        if hasattr(response, 'content'):
            response = response.content
        elif not isinstance(response, str):
            response = str(response)
        
        return response.strip()
    except Exception as e:
        # Return a user-friendly message instead of showing error
        return "I apologize, but I'm having trouble generating a response right now. Please try again or rephrase your question."

def save_to_vector_store(topic: str, explanation: str):
    """
    Save the topic and explanation to Chroma vector store for long-term memory
    This is non-blocking - returns immediately if vector store isn't ready
    """
    # Quick check - if embeddings aren't loaded, don't try
    if st.session_state.embeddings is None:
        return False  # Fail silently
    
    try:
        vector_store = initialize_vector_store()
        if vector_store is None:
            return False
        
        # Create a document with the topic and explanation
        doc = Document(
            page_content=f"Topic: {topic}\n\nExplanation: {explanation}",
            metadata={"topic": topic, "timestamp": datetime.now().isoformat()}
        )
        
        # Add to vector store (non-blocking operation)
        try:
            vector_store.add_documents([doc])
        except Exception:
            return False  # Fail silently
        
        return True
    except Exception:
        # Fail silently - don't slow down the app
        return False

def search_similar_topics(query: str, top_k: int = 3) -> List[Dict]:
    """
    Search for similar topics in the vector store
    """
    try:
        vector_store = initialize_vector_store()
        if vector_store is None:
            return []
        
        # Check if vector store has any documents
        try:
            # Try to get collection count (Chroma method)
            if hasattr(vector_store, '_collection'):
                collection = vector_store._collection
                if collection and hasattr(collection, 'count'):
                    count = collection.count()
                    if count == 0:
                        return []  # Empty store, return empty results
        except:
            pass  # If check fails, proceed with search anyway
        
        # Search for similar documents
        try:
            results = vector_store.similarity_search_with_score(query, k=top_k)
            
            return [
                {
                    "content": doc.page_content,
                    "score": score,
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                }
                for doc, score in results
            ]
        except Exception as search_error:
            # Try alternative search method
            try:
                if hasattr(vector_store, 'similarity_search'):
                    docs = vector_store.similarity_search(query, k=top_k)
                    return [
                        {
                            "content": doc.page_content,
                            "score": 0.0,  # No score available
                            "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                        }
                        for doc in docs
                    ]
            except:
                pass
            return []
    except Exception as e:
        # Silently return empty list if store has issues
        return []

def generate_quiz(topic: str, explanation: str = None) -> List[Dict]:
    """
    Generate 3-5 multiple choice questions about the topic
    """
    llm = st.session_state.llm
    if llm is None:
        return []
    
    # Use explanation if available, otherwise fetch from vector store
    if explanation is None:
        similar = search_similar_topics(topic, top_k=1)
        if similar:
            context = similar[0]["content"]
        else:
            context = f"Topic: {topic}"
    else:
        context = f"Topic: {topic}\n\nExplanation: {explanation}"
    
    # Simplified prompt for better generation with small models
    prompt = f"""Create 4 different multiple choice questions about: {topic}

Make each question unique and test different aspects.

Format as JSON:
{{
  "questions": [
    {{"question": "Question 1?", "options": ["A", "B", "C", "D"], "correct": 0}},
    {{"question": "Question 2?", "options": ["A", "B", "C", "D"], "correct": 1}},
    {{"question": "Question 3?", "options": ["A", "B", "C", "D"], "correct": 2}},
    {{"question": "Question 4?", "options": ["A", "B", "C", "D"], "correct": 0}}
  ]
}}

JSON only:"""
    
    try:
        # Use invoke() for newer LangChain versions, fallback to direct call for older versions
        try:
            response = llm.invoke(prompt)
        except AttributeError:
            # Older version - try direct call
            response = llm(prompt)
        
        # Handle response if it's a string or needs extraction
        if hasattr(response, 'content'):
            response = response.content
        elif not isinstance(response, str):
            response = str(response)
        
        # Try to extract JSON from response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
            response = response.strip()
        
        # Parse JSON
        quiz_data = json.loads(response)
        return quiz_data.get("questions", [])
    except json.JSONDecodeError:
        # Better fallback: create varied questions manually
        return [
            {
                "question": f"What is the main concept of {topic}?",
                "options": [
                    f"A core principle in {topic}",
                    "An unrelated concept",
                    "A different topic entirely",
                    "Something completely different"
                ],
                "correct": 0
            },
            {
                "question": f"How is {topic} typically used?",
                "options": [
                    "In practical applications",
                    "Only in theory",
                    "Rarely",
                    "Never"
                ],
                "correct": 0
            },
            {
                "question": f"What is important to remember about {topic}?",
                "options": [
                    "Key details matter",
                    "Nothing is important",
                    "Everything is the same",
                    "There are no details"
                ],
                "correct": 0
            }
        ]
    except Exception:
        # Silently fail - return empty list, app will handle gracefully
        return []

def save_notes():
    """Save the conversation history as notes to notes/my_notes.txt"""
    try:
        notes_dir = Path("./notes")
        notes_dir.mkdir(parents=True, exist_ok=True)
        
        notes_file = notes_dir / "my_notes.txt"
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build notes content
        notes_content = f"Study Notes - {timestamp}\n"
        notes_content += "=" * 50 + "\n\n"
        
        # Add conversation history
        for i, exchange in enumerate(st.session_state.conversation_history, 1):
            notes_content += f"Exchange {i}:\n"
            notes_content += f"Topic/Question: {exchange.get('topic', 'N/A')}\n"
            notes_content += f"User: {exchange.get('user', 'N/A')}\n"
            notes_content += f"AI: {exchange.get('assistant', 'N/A')}\n"
            notes_content += "-" * 30 + "\n\n"
        
        # Append to file (or create if doesn't exist)
        with open(notes_file, "a", encoding="utf-8") as f:
            f.write(notes_content)
        
        return True
    except Exception as e:
        st.error(f"Error saving notes: {str(e)}")
        return False

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="AI Study Buddy",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom dark theme CSS
    st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #fafafa;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1589cc;
    }
    h1, h2, h3 {
        color: #fafafa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and header
    st.title("📚 Personalized AI Study Buddy")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Directory setup
        if st.button("🔧 Initialize Directories"):
            create_directories()
            st.success("Directories initialized!")
        
        # Vector store diagnostics and reload
        st.markdown("---")
        st.subheader("🔍 Vector Store")
        
        if st.button("🔄 Reload Embeddings"):
            with st.spinner("Reloading embeddings..."):
                st.session_state.embeddings = None
                st.session_state.vector_store = None
                embeddings = load_embeddings()
                if embeddings:
                    initialize_vector_store()
                    st.success("✅ Embeddings reloaded!")
                else:
                    st.error("❌ Failed to reload embeddings.")
        
        if st.button("🧪 Test Vector Store"):
            vector_store = initialize_vector_store()
            if vector_store:
                try:
                    # Test saving
                    test_doc = Document(
                        page_content="Test document",
                        metadata={"test": True}
                    )
                    vector_store.add_documents([test_doc])
                    
                    # Test searching
                    results = vector_store.similarity_search("test", k=1)
                    st.success(f"✅ Vector store working! Found {len(results)} test result(s).")
                except Exception as e:
                    st.error(f"❌ Vector store test failed: {str(e)}")
            else:
                st.error("❌ Vector store not initialized. Load embeddings first.")
        
        # Model loading
        st.subheader("🤖 Model Setup")
        
        model_path = st.text_input(
            "Model Path (optional)",
            value="",
            help="Leave empty to auto-detect, or specify path to .gguf or .bin file"
        )
        
        if st.button("🔄 Load Model"):
            with st.spinner("Loading LLM..."):
                if model_path:
                    load_local_llm(model_path)
                else:
                    load_local_llm()
                if st.session_state.llm is not None:
                    st.success("Model loaded successfully!")
                else:
                    st.error("Failed to load model. Check the model path.")
        
        # Status indicators
        st.markdown("---")
        st.subheader("📊 Status")
        llm_status = "✅ Loaded" if st.session_state.llm is not None else "❌ Not Loaded"
        vector_status = "✅ Ready" if st.session_state.vector_store is not None else "❌ Not Ready"
        
        st.write(f"**LLM:** {llm_status}")
        st.write(f"**Vector Store:** {vector_status}")
        st.write(f"**Conversations:** {len(st.session_state.conversation_history)}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Study Session")
        
        # Display current topic if available
        if st.session_state.current_topic:
            st.info(f"📚 **Current Topic:** {st.session_state.current_topic}")
        
        # Topic input
        topic = st.text_input(
            "📖 Enter Topic or Question:",
            placeholder="e.g., 'What is machine learning?' or 'Explain quantum physics'",
            key="topic_input"
        )
        
        # Buttons row
        col_ask, col_quiz, col_notes, col_new = st.columns(4)
        
        with col_ask:
            ask_button = st.button("❓ Ask", use_container_width=True)
        
        with col_quiz:
            quiz_button = st.button("📝 Generate Quiz", use_container_width=True)
        
        with col_notes:
            save_notes_button = st.button("💾 Save Notes", use_container_width=True)
        
        with col_new:
            new_topic_button = st.button("🆕 New Topic", use_container_width=True)
        
        # Handle New Topic button - clear current topic to start fresh
        if new_topic_button:
            st.session_state.current_topic = ""
            st.success("✨ Ready for a new topic! Enter your question above.")
        
        # Handle Ask button
        if ask_button:
            if not topic.strip():
                st.warning("⚠️ Please enter a topic or question first!")
            elif st.session_state.llm is None:
                st.error("❌ Please load a model first from the sidebar!")
            else:
                # Show progress bar for long operations
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🤔 Processing your question...")
                    progress_bar.progress(20)
                    
                    # Determine if this is a new topic or follow-up question
                    current_topic = st.session_state.current_topic
                    user_input = topic.strip()
                    
                    progress_bar.progress(40)
                    status_text.text("🧠 Generating response...")
                    
                    # If no current topic, treat input as new topic
                    # If current topic exists, treat input as follow-up question
                    if not current_topic:
                        # New topic: use input as both topic and question
                        explanation = generate_explanation(user_input, None)
                        st.session_state.current_topic = user_input
                        topic_for_history = user_input
                    else:
                        # Follow-up: use current topic and input as question
                        explanation = generate_explanation(current_topic, user_input)
                        topic_for_history = current_topic
                    
                    progress_bar.progress(80)
                    status_text.text("💾 Saving...")
                    
                    # Add to conversation history (fast operation)
                    st.session_state.conversation_history.append({
                        "topic": topic_for_history,
                        "user": user_input,
                        "assistant": explanation,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Save to vector store (non-blocking, happens in background)
                    # Don't wait for this - it's optional and doesn't block UI
                    try:
                        save_to_vector_store(topic_for_history, explanation)
                    except:
                        pass  # Ignore errors to keep app fast
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Complete!")
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display response
                    st.markdown("### 💡 Explanation:")
                    st.markdown(explanation)
                except Exception:
                    # Hide errors from user - show friendly message instead
                    progress_bar.empty()
                    status_text.empty()
                    st.info("💡 Please try rephrasing your question or check that your model is loaded correctly.")
        
        # Handle Generate Quiz button - only generates, doesn't display
        if quiz_button:
            if not st.session_state.current_topic:
                st.warning("⚠️ Please ask a question first to establish a topic!")
            elif st.session_state.llm is None:
                st.error("❌ Please load a model first from the sidebar!")
            else:
                with st.spinner("📝 Generating quiz..."):
                    try:
                        # Get the most recent explanation
                        recent_explanation = None
                        if st.session_state.conversation_history:
                            recent_explanation = st.session_state.conversation_history[-1].get("assistant")
                        
                        quiz_questions = generate_quiz(st.session_state.current_topic, recent_explanation)
                        
                        if quiz_questions:
                            # Store quiz in session state
                            if "quiz_data" not in st.session_state:
                                st.session_state.quiz_data = []
                            if "quiz_answers" not in st.session_state:
                                st.session_state.quiz_answers = {}
                            
                            # Clear previous quiz state when generating new quiz
                            st.session_state.quiz_data = quiz_questions
                            st.session_state.quiz_answers = {}
                            
                            # Clear all individual answer keys
                            for i in range(len(quiz_questions)):
                                answer_key = f"quiz_answer_{i}"
                                if answer_key in st.session_state:
                                    del st.session_state[answer_key]
                            
                            st.success("✅ Quiz generated! Scroll down to see questions.")
                        else:
                            # No questions generated - show friendly message
                            st.info("📝 Quiz generation is taking longer than expected. Please try again in a moment.")
                    except Exception:
                        # Hide errors - show friendly message
                        st.info("📝 Having trouble generating quiz questions. Please try asking a question first, then generating a quiz.")
        
        # Display quiz from session state (persists across reruns)
        if "quiz_data" in st.session_state and st.session_state.quiz_data:
            quiz_questions = st.session_state.quiz_data
            
            st.markdown("---")
            st.markdown("### 📝 Quiz Questions")
            st.markdown("Select your answers and click 'Check Answers' at the bottom!")
            st.markdown("---")
            
            # Display interactive questions
            for i, q in enumerate(quiz_questions):
                question_text = q.get("question", "No question text")
                options = q.get("options", [])
                
                # Remove any emoji markers from options if present
                clean_options = [opt.replace("✅", "").replace("○", "").strip() for opt in options]
                
                st.markdown(f"**Question {i+1}:** {question_text}")
                
                # Use radio buttons for selection
                answer_key = f"quiz_answer_{i}"
                radio_key = f"quiz_radio_{i}"
                
                # Initialize answer key if not exists
                if answer_key not in st.session_state:
                    st.session_state[answer_key] = None
                
                # Get current selection
                current_selection = st.session_state[answer_key]
                
                # Radio buttons with index-based selection
                try:
                    selected_index = st.radio(
                        "Choose an answer:",
                        range(len(clean_options)),
                        format_func=lambda x: clean_options[x],
                        key=radio_key,
                        label_visibility="collapsed",
                        index=current_selection if current_selection is not None and current_selection < len(clean_options) else None
                    )
                    
                    # Always update the state with current selection
                    st.session_state[answer_key] = selected_index
                    st.session_state.quiz_answers[i] = selected_index
                except Exception:
                    # Fallback if radio fails - silently continue
                    selected_index = None
                
                st.markdown("---")
            
            # Add button to check answers (separate from quiz generation)
            check_button = st.button("✅ Check Answers", key="check_quiz")
            
            if check_button:
                correct_count = 0
                total = len(quiz_questions)
                
                st.markdown("### 📊 Results:")
                
                for i, q in enumerate(quiz_questions):
                    correct_idx = q.get("correct", 0)
                    answer_key = f"quiz_answer_{i}"
                    user_answer = st.session_state.get(answer_key, -1)
                    
                    # Fallback to quiz_answers dict if needed
                    if user_answer == -1:
                        user_answer = st.session_state.quiz_answers.get(i, -1)
                    
                    options = q.get("options", [])
                    clean_options = [opt.replace("✅", "").replace("○", "").strip() for opt in options]
                    
                    is_correct = user_answer == correct_idx and user_answer >= 0
                    if is_correct:
                        correct_count += 1
                    
                    # Show result for each question
                    result_emoji = "✅" if is_correct else "❌"
                    st.markdown(f"**Q{i+1}:** {result_emoji} {q.get('question', '')}")
                    
                    if user_answer >= 0 and user_answer < len(clean_options):
                        user_answer_text = clean_options[user_answer]
                    else:
                        user_answer_text = "Not answered"
                    
                    if not is_correct:
                        st.markdown(f"   Your answer: {user_answer_text}")
                        if correct_idx < len(clean_options):
                            st.markdown(f"   Correct answer: **{clean_options[correct_idx]}**")
                        else:
                            st.markdown(f"   Correct answer: **Option {correct_idx + 1}**")
                
                # Show overall score
                score_percent = int((correct_count / total) * 100) if total > 0 else 0
                st.markdown("---")
                st.markdown(f"### 🎯 Score: {correct_count}/{total} ({score_percent}%)")
                
                if score_percent == 100:
                    st.success("🎉 Perfect score! Excellent work!")
                elif score_percent >= 70:
                    st.info("👍 Good job! Keep studying!")
                else:
                    st.warning("📚 Review the material and try again!")
                
                # Show correct answers
                with st.expander("📖 Show All Correct Answers"):
                    for i, q in enumerate(quiz_questions):
                        correct_idx = q.get("correct", 0)
                        options = q.get("options", [])
                        clean_options = [opt.replace("✅", "").replace("○", "").strip() for opt in options]
                        st.markdown(f"**Q{i+1}:** {q.get('question', '')}")
                        st.markdown(f"   ✓ {clean_options[correct_idx]}")
                        st.markdown("")
        
        # Handle Save Notes button
        if save_notes_button:
            if not st.session_state.conversation_history:
                st.info("📝 No conversation history to save yet. Ask some questions first!")
            else:
                with st.spinner("💾 Saving notes..."):
                    try:
                        if save_notes():
                            st.success(f"✅ Notes saved to ./notes/my_notes.txt")
                        else:
                            st.info("💡 Notes are being saved. Please try again if needed.")
                    except Exception:
                        st.info("💡 Notes saving is in progress. Please check the ./notes/ folder.")
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.markdown("---")
            st.header("📜 Conversation History")
            
            for i, exchange in enumerate(reversed(st.session_state.conversation_history), 1):
                with st.expander(f"Exchange {len(st.session_state.conversation_history) - i + 1}: {exchange.get('user', 'N/A')[:50]}..."):
                    st.markdown(f"**Topic:** {exchange.get('topic', 'N/A')}")
                    st.markdown(f"**Your Question:** {exchange.get('user', 'N/A')}")
                    st.markdown(f"**AI Response:** {exchange.get('assistant', 'N/A')}")
    
    with col2:
        st.header("🔍 Similar Topics")
        
        if st.session_state.current_topic:
            search_query = st.text_input(
                "Search related topics:",
                value=st.session_state.current_topic,
                key="search_input"
            )
            
            if st.button("🔎 Search", key="search_btn"):
                if search_query:
                    with st.spinner("Searching..."):
                        similar = search_similar_topics(search_query, top_k=3)
                        
                        if similar:
                            for item in similar:
                                with st.expander(f"Similarity: {1 - item['score']:.2%}"):
                                    st.markdown(item["content"][:300] + "...")
                        else:
                            st.info("No similar topics found yet.")
        else:
            st.info("Ask a question first to enable topic search.")

if __name__ == "__main__":
    # Initialize directories on startup (fast, non-blocking)
    create_directories()
    
    # Don't load embeddings on startup - load lazily when needed
    # This makes the app start much faster
    # Embeddings will load when:
    # 1. User clicks "Reload Embeddings" in sidebar
    # 2. User tries to use vector store features
    
    # Run the main app immediately
    main()

