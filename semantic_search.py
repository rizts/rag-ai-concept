import os
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import numpy as np
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# Initialize HuggingFace Inference Client
HF_TOKEN = os.getenv("HF_API_KEY")
client = InferenceClient(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", token=HF_TOKEN)

class ConversationMemory:
    """Manage conversation history for multi-turn RAG"""
    
    def __init__(self, max_history: int = 5):
        self.history: List[Dict] = []
        self.max_history = max_history
    
    def add_turn(self, query: str, answer: str, sources: List[dict]):
        """Add a conversation turn"""
        self.history.append({
            "query": query,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last N turns
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self) -> str:
        """Get conversation history as context"""
        if not self.history:
            return ""
        
        context_parts = ["Previous conversation:"]
        for i, turn in enumerate(self.history, 1):
            context_parts.append(f"\nQ{i}: {turn['query']}")
            context_parts.append(f"A{i}: {turn['answer'][:200]}...")
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear conversation history"""
        self.history = []
    
    def export(self, filepath: str):
        """Export conversation to JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "conversation": self.history,
                "total_turns": len(self.history),
                "exported_at": datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)

# ========== EMBEDDINGS ==========

def get_embedding(text: str) -> list[float]:
    """Get embedding from HuggingFace"""
    response = client.feature_extraction(text)
    # The response is a numpy array, convert to list
    if hasattr(response, 'tolist'):
        return response.tolist()
    else:
        # If it's already a list or array-like
        return list(response)


def get_query_embedding(text: str) -> list[float]:
    """Get embedding for query"""
    response = client.feature_extraction(text)
    # The response is a numpy array, convert to list
    if hasattr(response, 'tolist'):
        return response.tolist()
    else:
        # If it's already a list or array-like
        return list(response)

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ========== TEXT PROCESSING ==========

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
    
    return chunks if chunks else [text]

def highlight_text(text: str, query: str, context_chars: int = 100) -> str:
    """Highlight query terms in text with context"""
    query_words = query.lower().split()
    text_lower = text.lower()
    
    best_pos = -1
    for word in query_words:
        pos = text_lower.find(word)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos
    
    if best_pos == -1:
        return text[:context_chars * 2] + "..."
    
    start = max(0, best_pos - context_chars)
    end = min(len(text), best_pos + context_chars)
    
    snippet = text[start:end]
    
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    
    return snippet

def keyword_score(text: str, query: str) -> float:
    """Calculate keyword match score"""
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    
    if not query_words:
        return 0.0
    
    matches = query_words.intersection(text_words)
    return len(matches) / len(query_words)

# ========== DOCUMENT LOADING ==========

def load_documents(folder: Path) -> List[dict]:
    """Load and chunk documents with metadata"""
    files = list(folder.glob("**/*.txt")) + list(folder.glob("**/*.md"))
    
    docs = []
    for f in files:
        try:
            text = f.read_text(encoding='utf-8')
            chunks = chunk_text(text)
            
            stat = f.stat()
            file_ext = f.suffix.lower()
            
            for i, chunk in enumerate(chunks):
                docs.append({
                    "filename": f.name,
                    "chunk_id": i,
                    "path": str(f),
                    "text": chunk,
                    "metadata": {
                        "file_type": file_ext,
                        "file_size": stat.st_size,
                        "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "word_count": len(chunk.split()),
                        "total_chunks": len(chunks)
                    }
                })
        except Exception as e:
            print(f"Error loading {f.name}: {e}")
    
    return docs

# ========== INDEXING ==========

def create_index(docs: List[dict]) -> List[dict]:
    """Create embeddings for all documents"""
    print(f"Creating embeddings for {len(docs)} documents...")
    
    for i, doc in enumerate(docs, 1):
        chunk_label = f" (chunk {doc['chunk_id']})" if doc['chunk_id'] > 0 else ""
        print(f"  [{i}/{len(docs)}] {doc['filename']}{chunk_label}")
        doc['embedding'] = get_embedding(doc['text'])
    
    return docs

def save_index(docs: List[dict], filepath: str = "index_gemini.json"):
    """Save index to JSON"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Index saved to {filepath}")

def load_index(filepath: str = "index_gemini.json") -> List[dict]:
    """Load index from JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# ========== FILTERING ==========

def filter_by_metadata(docs: List[dict], 
                       filename: Optional[str] = None,
                       file_type: Optional[str] = None,
                       min_words: Optional[int] = None) -> List[dict]:
    """Filter documents by metadata"""
    filtered = docs
    
    if filename:
        filtered = [d for d in filtered if filename.lower() in d['filename'].lower()]
    
    if file_type:
        filtered = [d for d in filtered if d['metadata']['file_type'] == file_type]
    
    if min_words:
        filtered = [d for d in filtered if d['metadata']['word_count'] >= min_words]
    
    return filtered

# ========== SEARCH ==========

def hybrid_search(query: str, docs: List[dict], 
                  top_k: int = 3,
                  semantic_weight: float = 0.7,
                  keyword_weight: float = 0.3) -> List[dict]:
    """Hybrid search: semantic + keyword"""
    print(f"\n🔍 Hybrid search for: '{query}'")
    
    query_emb = get_query_embedding(query)
    
    for doc in docs:
        semantic_score = cosine_similarity(query_emb, doc['embedding'])
        kw_score = keyword_score(doc['text'], query)
        
        doc['semantic_score'] = semantic_score
        doc['keyword_score'] = kw_score
        doc['score'] = (semantic_weight * semantic_score) + (keyword_weight * kw_score)
    
    results = sorted(docs, key=lambda x: x['score'], reverse=True)[:top_k]
    return results

# ========== RAG ==========

def generate_answer(query: str, context_chunks: List[str]) -> dict:
    """Generate answer using HuggingFace with retrieved context"""
    
    context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)])
    
    prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say "I cannot find the answer in the provided documents."

Context:
{context}

Question: {query}

Answer:"""
    
    # Note: This function still uses HuggingFace for text generation, only embeddings changed
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    
    return {
        "answer": response.text,
        "context_used": len(context_chunks),
        "prompt_tokens": len(prompt.split())
    }

def generate_answer_with_history(query: str, 
                                 context_chunks: List[str],
                                 conversation_history: str = "") -> dict:
    """Generate answer with conversation context"""
    
    # Combine retrieved context
    context = "\n\n".join([f"[Document {i+1}]\n{chunk}" 
                          for i, chunk in enumerate(context_chunks)])
    
    # Build prompt with conversation history
    if conversation_history:
        prompt = f"""{conversation_history}

---

Based on the conversation above and the following documents, answer the new question.

Documents:
{context}

New Question: {query}

Instructions:
- Reference previous conversation if relevant
- Use the documents to provide accurate information
- If the answer requires information from both documents and previous conversation, combine them
- If you cannot answer from the provided information, say so clearly

Answer:"""
    else:
        prompt = f"""Based on the following documents, answer the question.

Documents:
{context}

Question: {query}

Instructions:
- Provide a clear, accurate answer based on the documents
- If the answer is not in the documents, say "I cannot find the answer in the provided documents"

Answer:"""
    
    # Note: This function still uses HuggingFace for text generation, only embeddings changed
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    
    return {
        "answer": response.text,
        "context_used": len(context_chunks),
        "prompt_tokens": len(prompt.split()),
        "has_history": bool(conversation_history)
    }

import time

def generate_answer_streaming(query: str, 
                              context_chunks: List[str],
                              conversation_history: str = "",
                              typing_speed: float = 0.02):  # Delay per char
    """Generate answer with streaming and typing effect"""
    
    context = "\n\n".join([f"[Document {i+1}]\n{chunk}" 
                          for i, chunk in enumerate(context_chunks)])
    
    if conversation_history:
        prompt = f"""{conversation_history}

---

Based on the conversation above and the following documents, answer the new question.

Documents:
{context}

New Question: {query}

Instructions:
- Reference previous conversation if relevant
- Use the documents to provide accurate information
- If the answer requires information from both documents and previous conversation, combine them
- If you cannot answer from the provided information, say so clearly

Answer:"""
    else:
        prompt = f"""Based on the following documents, answer the question.

Documents:
{context}

Question: {query}

Instructions:
- Provide a clear, accurate answer based on the documents
- If the answer is not in the documents, say "I cannot find the answer in the provided documents"

Answer:"""
    
    # Note: This function still uses HuggingFace for text generation, only embeddings changed
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt, stream=True)
    
    full_answer = ""
    for chunk in response:
        if chunk.text:
            full_answer += chunk.text
            
            # Stream character by character with delay
            for char in chunk.text:
                print(char, end='', flush=True)
                time.sleep(typing_speed)  # Typing effect
    
    yield {
        "done": True,
        "full_answer": full_answer,
        "context_used": len(context_chunks),
        "prompt_tokens": len(prompt.split()),
        "has_history": bool(conversation_history)
    }

def rag_query(query: str, 
                       docs: List[dict],
                       conversation: ConversationMemory,
                       top_k: int = 3,
                       use_hybrid: bool = True,
                       rewrite: bool = True) -> dict:
    """Advanced RAG with conversation memory and query rewriting"""
    
    print(f"\n🤖 Advanced RAG Query: '{query}'")
    print("="*60)
    
    # Get conversation context
    conv_history = conversation.get_context()
    
    # Step 1: Query rewriting (if follow-up question)
    original_query = query
    if rewrite and conv_history:
        print("🔄 Step 1: Rewriting query with conversation context...")
        query = rewrite_query(query, conv_history)
        print(f"   Original: {original_query}")
        print(f"   Rewritten: {query}\n")
    else:
        print("📝 Step 1: Using original query (no rewriting needed)\n")
    
    # Step 2: Retrieve relevant documents
    print("📚 Step 2: Retrieving relevant documents...")
    
    if use_hybrid:
        results = hybrid_search(query, docs, top_k=top_k)
    else:
        query_emb = get_query_embedding(query)
        for doc in docs:
            doc['score'] = cosine_similarity(query_emb, doc['embedding'])
        results = sorted(docs, key=lambda x: x['score'], reverse=True)[:top_k]
    
    context_chunks = [r['text'] for r in results]
    sources = [{"filename": r['filename'], "chunk_id": r['chunk_id'], "score": r['score']} 
               for r in results]
    
    print(f"✅ Retrieved {len(results)} relevant chunks\n")
    
    # Step 3: Generate answer with conversation context
    print("💭 Step 3: Generating answer with conversation context...")
    answer_data = generate_answer_with_history(original_query, context_chunks, conv_history)
    
    print("✅ Answer generated\n")
    
    # Add to conversation history
    conversation.add_turn(original_query, answer_data['answer'], sources)
    
    return {
        "query": original_query,
        "rewritten_query": query if rewrite and conv_history else None,
        "answer": answer_data['answer'],
        "sources": sources,
        "context_chunks": context_chunks,
        "metadata": {
            "chunks_used": len(context_chunks),
            "prompt_tokens": answer_data['prompt_tokens'],
            "has_conversation_history": answer_data['has_history'],
            "conversation_turns": len(conversation.history)
        }
    }

def rag_query_streaming(query: str, 
                       docs: List[dict],
                       conversation: ConversationMemory,
                       top_k: int = 3,
                       use_hybrid: bool = True,
                       rewrite: bool = True):
    """Advanced RAG with streaming response"""
    
    print(f"\n🤖 Streaming RAG Query: '{query}'")
    print("="*60)
    
    # Get conversation context
    conv_history = conversation.get_context()
    
    # Step 1: Query rewriting
    original_query = query
    if rewrite and conv_history:
        print("🔄 Step 1: Rewriting query...")
        query = rewrite_query(query, conv_history)
        print(f"   Original: {original_query}")
        print(f"   Rewritten: {query}\n")
    else:
        print("📝 Step 1: Using original query\n")
    
    # Step 2: Retrieve
    print("📚 Step 2: Retrieving documents...")
    
    if use_hybrid:
        results = hybrid_search(query, docs, top_k=top_k)
    else:
        query_emb = get_query_embedding(query)
        for doc in docs:
            doc['score'] = cosine_similarity(query_emb, doc['embedding'])
        results = sorted(docs, key=lambda x: x['score'], reverse=True)[:top_k]
    
    context_chunks = [r['text'] for r in results]
    sources = [{"filename": r['filename'], "chunk_id": r['chunk_id'], "score": r['score']} 
               for r in results]
    
    print(f"✅ Retrieved {len(results)} chunks\n")
    
    # Step 3: Stream answer
    print("💭 Step 3: Generating answer...\n")
    print("="*60)
    print("💡 ANSWER")
    print("="*60)
    
    # Stream and collect answer
    full_answer = ""
    metadata = {}
    
    for chunk in generate_answer_streaming(original_query, context_chunks, conv_history):
        if isinstance(chunk, dict) and chunk.get("done"):
            # Final metadata
            full_answer = chunk['full_answer']
            metadata = {
                "chunks_used": chunk['context_used'],
                "prompt_tokens": chunk['prompt_tokens'],
                "has_conversation_history": chunk['has_history'],
                "conversation_turns": len(conversation.history)
            }
        else:
            # Stream text token
            print(chunk, end='', flush=True)
    
    print("\n" + "="*60)
    
    # Add to conversation
    conversation.add_turn(original_query, full_answer, sources)
    
    # Return result
    result = {
        "query": original_query,
        "rewritten_query": query if rewrite and conv_history else None,
        "answer": full_answer,
        "sources": sources,
        "context_chunks": context_chunks,
        "metadata": metadata
    }
    
    # Display sources
    print("\n📚 SOURCES")
    print("="*60)
    for i, source in enumerate(sources, 1):
        chunk_info = f" [chunk {source['chunk_id']}]" if source['chunk_id'] > 0 else ""
        print(f"{i}. {source['filename']}{chunk_info} (relevance: {source['score']:.3f})")
    
    print("\n" + "="*60)
    print(f"📊 Metadata: {metadata['chunks_used']} chunks, "
          f"~{metadata['prompt_tokens']} tokens, "
          f"{metadata['conversation_turns']} turns")
    print("="*60)
    
    return result

# ========== DISPLAY ==========

def display_results(results: List[dict], query: str, show_scores: bool = True):
    """Display search results with highlighting"""
    print(f"\n📄 Top {len(results)} results:\n")
    
    for i, r in enumerate(results, 1):
        chunk_info = f" [chunk {r['chunk_id']}]" if r['chunk_id'] > 0 else ""
        
        print(f"{i}. {r['filename']}{chunk_info}")
        
        if show_scores:
            print(f"   Score: {r['score']:.3f} ", end="")
            if 'semantic_score' in r:
                print(f"(semantic: {r['semantic_score']:.3f}, keyword: {r['keyword_score']:.3f})")
            else:
                print()
        
        snippet = highlight_text(r['text'], query, context_chars=150)
        print(f"   {snippet}")
        
        print(f"   📊 {r['metadata']['word_count']} words, modified: {r['metadata']['modified_date'][:10]}\n")

def display_rag_result(result: dict, show_context: bool = False):
    """Display advanced RAG result"""
    
    print("="*60)
    print("❓ QUESTION")
    print("="*60)
    print(result['query'])
    
    if result.get('rewritten_query'):
        print(f"\n🔄 Rewritten as: {result['rewritten_query']}")
    
    print("\n" + "="*60)
    print("💡 ANSWER")
    print("="*60)
    print(result['answer'])
    
    print("\n" + "="*60)
    print("📚 SOURCES")
    print("="*60)
    for i, source in enumerate(result['sources'], 1):
        chunk_info = f" [chunk {source['chunk_id']}]" if source['chunk_id'] > 0 else ""
        print(f"{i}. {source['filename']}{chunk_info} (relevance: {source['score']:.3f})")
    
    if show_context:
        print("\n" + "="*60)
        print("📖 CONTEXT USED")
        print("="*60)
        for i, chunk in enumerate(result['context_chunks'], 1):
            print(f"\n[{i}] {chunk[:300]}...")
    
    print("\n" + "="*60)
    meta = result['metadata']
    print(f"📊 Metadata: {meta['chunks_used']} chunks, "
          f"~{meta['prompt_tokens']} tokens, "
          f"{meta['conversation_turns']} turns in history")
    print("="*60)

# ========== EXPORT ==========

def export_results(results: List[dict], query: str, format: str = "json"):
    """Export search results to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "json":
        filename = f"results_{timestamp}.json"
        
        export_data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "total_results": len(results),
            "results": [
                {
                    "filename": r['filename'],
                    "chunk_id": r['chunk_id'],
                    "score": r['score'],
                    "semantic_score": r.get('semantic_score', 0),
                    "keyword_score": r.get('keyword_score', 0),
                    "text": r['text'],
                    "metadata": r['metadata']
                }
                for r in results
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    elif format == "csv":
        import csv
        filename = f"results_{timestamp}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Chunk', 'Score', 'Semantic', 'Keyword', 'Text'])
            
            for r in results:
                writer.writerow([
                    r['filename'],
                    r['chunk_id'],
                    f"{r['score']:.3f}",
                    f"{r.get('semantic_score', 0):.3f}",
                    f"{r.get('keyword_score', 0):.3f}",
                    r['text'][:200]
                ])
    
    print(f"💾 Results exported to: {filename}")
    return filename

# ========== REWRITE QUERY ==========

def rewrite_query(query: str, conversation_history: str) -> str:
    """Rewrite query to be standalone using conversation context"""
    
    if not conversation_history:
        return query
    
    prompt = f"""Given the conversation history, rewrite the follow-up question to be a standalone question that includes necessary context.

{conversation_history}

Follow-up question: {query}

Standalone question:"""
    
    # Note: This function still uses HuggingFace for text generation, only embeddings changed
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    
    rewritten = response.text.strip()
    return rewritten

# ========== INTERACTIVE ==========

def interactive_search(docs: List[dict]):
    """Interactive search with streaming support"""
    
    print("\n" + "="*60)
    print("🔍 ADVANCED RAG SYSTEM - HuggingFace Embeddings (with Streaming)")
    print("="*60)
    print("\nCommands:")
    print("  ask <query>          - RAG with conversation (non-streaming)")
    print("  stream <query>       - RAG with streaming response ⚡")
    print("  rag <query>          - Same as 'ask'")
    print("  search <query>       - Semantic search only")
    print("  hybrid <query>       - Hybrid search")
    print("  new                  - Start new conversation")
    print("  history              - Show conversation history")
    print("  save                 - Save conversation to file")
    print("  filter <filename>    - Filter documents")
    print("  stats                - Show statistics")
    print("  help                 - Show this help")
    print("  quit                 - Exit")
    print("="*60)
    
    conversation = ConversationMemory(max_history=5)
    
    typing_speed = 0.02  # Default

    last_results = []
    last_query = ""
    filtered_docs = docs
    
    while True:
        cmd = input("\n> ").strip()
        
        if not cmd:
            continue
        
        if cmd.lower() in ['quit', 'exit', 'q']:
            if conversation.history:
                save = input("\n💾 Save conversation? (y/n): ")
                if save.lower() == 'y':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"conversation_{timestamp}.json"
                    conversation.export(filename)
                    print(f"✅ Saved to {filename}")
            break
        
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command in ["ask", "rag"]:
            if not args:
                print(f"❌ Usage: {command} <query>")
                continue
            
            result = rag_query(
                args, 
                filtered_docs, 
                conversation,
                top_k=3, 
                use_hybrid=True,
                rewrite=True
            )
            display_rag_result(result, show_context=False)
            last_query = args
        
        elif command == "stream":
            if not args:
                print("❌ Usage: stream <query>")
                continue
            
            # Streaming RAG
            result = rag_query_streaming(
                args,
                filtered_docs,
                conversation,
                top_k=3,
                use_hybrid=True,
                rewrite=True
            )
            last_query = args
        
        elif command == "search":
            if not args:
                print("❌ Usage: search <query>")
                continue
            
            results = hybrid_search(args, filtered_docs, top_k=5, 
                                   semantic_weight=1.0, keyword_weight=0.0)
            display_results(results, args)
            last_results = results
            last_query = args
        
        elif command == "hybrid":
            if not args:
                print("❌ Usage: hybrid <query>")
                continue
            
            results = hybrid_search(args, filtered_docs, top_k=5)
            display_results(results, args)
            last_results = results
            last_query = args
        
        elif command == "new":
            conversation.clear()
            print("✅ New conversation started")
        
        elif command == "history":
            if not conversation.history:
                print("ℹ️  No conversation history")
            else:
                print(f"\n💬 Conversation History ({len(conversation.history)} turns):")
                print("="*60)
                for i, turn in enumerate(conversation.history, 1):
                    print(f"\n[Turn {i}] {turn['timestamp'][:19]}")
                    print(f"Q: {turn['query']}")
                    print(f"A: {turn['answer'][:150]}...")
                print("="*60)
        
        elif command == "save":
            if not conversation.history:
                print("❌ No conversation to save")
                continue
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
            conversation.export(filename)
            print(f"✅ Saved to {filename}")
        
        elif command == "filter":
            if not args:
                filtered_docs = docs
                print(f"✅ Filter cleared. {len(docs)} documents available")
            else:
                filtered_docs = filter_by_metadata(docs, filename=args)
                print(f"✅ Filtered to {len(filtered_docs)} documents matching '{args}'")
        
        elif command == "stats":
            total_docs = len(set(d['filename'] for d in docs))
            total_chunks = len(docs)
            file_types = {}
            
            for d in docs:
                ft = d['metadata']['file_type']
                file_types[ft] = file_types.get(ft, 0) + 1
            
            print(f"\n📊 System Statistics:")
            print(f"   Total files: {total_docs}")
            print(f"   Total chunks: {total_chunks}")
            print(f"   File types: {dict(file_types)}")
            print(f"   Filtered docs: {len(filtered_docs)} chunks")
            print(f"   Conversation turns: {len(conversation.history)}")
        
        elif command == "help":
            print("\n" + "="*60)
            print("🔍 ADVANCED RAG SYSTEM - Commands (HuggingFace Embeddings)")
            print("="*60)
            print("\nMain Commands:")
            print("  ask/rag <query>      - Ask question (wait for full answer)")
            print("  stream <query>       - Ask with streaming ⚡ (see answer appear)")
            print("  search <query>       - Search only, no answer")
            print("  hybrid <query>       - Hybrid search (semantic + keyword)")
            print("\nConversation:")
            print("  new                  - Clear conversation")
            print("  history              - View history")
            print("  save                 - Export to JSON")
            print("\nFiltering:")
            print("  filter <filename>    - Filter by filename")
            print("  filter               - Clear filter")
            print("\nInfo:")
            print("  stats                - Statistics")
            print("  help                 - This help")
            print("  quit                 - Exit")
            print("="*60)
        
        else:
            print(f"❌ Unknown command: {command}")
            print("   Type 'help' for commands")

# ========== MAIN ==========

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced RAG System (HuggingFace Embeddings)")
    parser.add_argument("folder", type=Path, help="Documents folder")
    parser.add_argument("--index", action="store_true", help="Create new index")
    args = parser.parse_args()
    
    index_file = "index_hf.json"
    
    if args.index or not os.path.exists(index_file):
        print("📚 Loading documents...")
        docs = load_documents(args.folder)
        print(f"Found {len(docs)} document chunks\n")
        
        docs = create_index(docs)
        save_index(docs, index_file)
    else:
        print(f"📂 Loading index from {index_file}...")
        docs = load_index(index_file)
        print(f"✅ Loaded {len(docs)} document chunks")
    
    # Use advanced interactive mode
    interactive_search(docs)

if __name__ == "__main__":
    main()