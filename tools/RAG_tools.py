"""
RAG tools for vehicle assistant - Complete integration from version3_refactor.py
This module contains the @tool-decorated helper functions used by the diagnostic workflow.
"""

import os
import re
import json
import logging
import requests
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from pathlib import Path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------
# Paths and Setup
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_DB_DIR = BASE_DIR / "output" / "chroma_db"

if not CHROMA_DB_DIR.exists() or not any(CHROMA_DB_DIR.iterdir()):
    logger.error("❌ Chroma DB not found. Please run RAG setup first.")
    # Don't exit, just log error - let the tools handle gracefully

# -------------------------
# Initialize LLM + embeddings + retriever
# -------------------------
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing - set it in .env")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

try:
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embeddings,
        collection_name="pdf_chunks",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
except Exception as e:
    logger.error(f"Failed to initialize Chroma vectorstore: {e}")
    vectorstore = None
    retriever = None

# Retrieval grader setup
grader_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=OPENAI_API_KEY)
grader_prompt = PromptTemplate(
    template="""You are a teacher grading a quiz. You will be given: 1/ a QUESTION 2/ A FACT provided by the student
You are grading RELEVANCE RECALL: A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION. A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION.
Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
Question: {question} Fact: {documents}""",
    input_variables=["question", "documents"],
)
retrieval_grader = grader_prompt | grader_llm | JsonOutputParser()

# Global NLP model (lazy-loaded)
_nlp_model = None

def _get_nlp():
    global _nlp_model
    if _nlp_model is None:
        import spacy
        try:
            _nlp_model = spacy.load("en_core_web_trf")
        except Exception:
            try:
                _nlp_model = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning(f"Could not load spacy model: {e}")
                _nlp_model = None
    return _nlp_model

# -------------------------
# TOOL FUNCTIONS (Complete from version3_refactor.py)
# -------------------------


@tool
def is_vehicle_related(question: str) -> dict:
    """Check if the question is vehicle-related before processing."""
    classifier_prompt = f"""
You are a classifier. Decide if the user question is about vehicle diagnostics, repair, or automotive problems.
Answer YES if it's about vehicle issues, maintenance, repairs, faults, or checks.
Answer NO if it's unrelated to vehicles.
Examples:
- "How do I replace my brake pads?" → YES
- "What is the capital of France?" → NO
- "Can I change the engine oil myself?" → YES
- "Tell me a joke." → NO
QUESTION: {question}
Answer only with "YES" or "NO".
"""
    try:
        response = llm.invoke(classifier_prompt)
        is_related = response.content.strip().upper() == "YES"
        logger.info(f"Vehicle relation check: {question[:50]}... → {is_related}")
        return {
            "is_vehicle_related": is_related,
            "message": "Vehicle-related question detected" if is_related else "Not vehicle-related",
        }
    except Exception as e:
        logger.error(f"Error in vehicle relation check: {e}")
        return {"is_vehicle_related": True, "message": "Error in classification, defaulting to vehicle-related"}

@tool
def extract_vehicle_model(question: str) -> dict:
    """Extract vehicle make and model from the question using NLP."""
    nlp_model = _get_nlp()
    if not nlp_model:
        logger.warning("NLP model not available, falling back to simple extraction")
        # Simple fallback extraction
        words = question.split()
        potential_vehicle = []
        for i, word in enumerate(words):
            if word.lower() in ['toyota', 'honda', 'ford', 'bmw', 'mercedes', 'audi', 'volkswagen', 'nissan', 'hyundai', 'kia']:
                potential_vehicle.append(word.title())
                if i + 1 < len(words):
                    potential_vehicle.append(words[i + 1].title())
                break
        
        if potential_vehicle:
            vehicle_info = " ".join(potential_vehicle[:2])
            return {"vehicle_info": vehicle_info, "found": True}
        else:
            return {"vehicle_info": None, "found": False}
    
    doc = nlp_model(question)
    entities = []
    
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            model_tokens = [ent.text]
            next_token = ent.end
            while next_token < len(doc) and (
                doc[next_token].is_title or doc[next_token].like_num or doc[next_token].is_lower
            ):
                model_tokens.append(doc[next_token].text)
                next_token += 1
            entities.append(" ".join(model_tokens))

    if entities:
        vehicle_info = entities[0].title()
    else:
        tokens = [t for t in doc if not t.is_stop and t.pos_ in ["PROPN", "NUM", "NOUN"]]
        if len(tokens) >= 2:
            model_tokens = []
            for t in tokens:
                if t.pos_ in ["PROPN", "NUM"]:
                    model_tokens.append(t.text)
                else:
                    break
            vehicle_info = " ".join(model_tokens).title() if model_tokens else None
        else:
            vehicle_info = None

    logger.info(f"Vehicle extraction: {question[:50]}... → {vehicle_info}")
    return {"vehicle_info": vehicle_info, "found": vehicle_info is not None}

def normalize_dtc_codes(text: str) -> str:
    """
    Normalize spoken DTC codes to standard format - ULTRA FLEXIBLE VERSION.
    
    Examples:
    - "P zero three zero one" → "P0301"
    - "tell me about p zero three zero one" → "tell me about P0301"
    """
    
    # Dictionary to convert word numbers to digits
    word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }
    
    # ULTRA FLEXIBLE pattern - allows any non-word chars between parts
    pattern = r'\b([PBCUpbcu])\W*(zero|one|two|three|four|five|six|seven|eight|nine)\W*(zero|one|two|three|four|five|six|seven|eight|nine)\W*(zero|one|two|three|four|five|six|seven|eight|nine)\W*(zero|one|two|three|four|five|six|seven|eight|nine)\b'
    
    def replace_dtc(match):
        prefix = match.group(1).upper()  # P, B, C, or U
        digit1 = word_to_digit[match.group(2).lower()]
        digit2 = word_to_digit[match.group(3).lower()]
        digit3 = word_to_digit[match.group(4).lower()]
        digit4 = word_to_digit[match.group(5).lower()]
        
        result = f"{prefix}{digit1}{digit2}{digit3}{digit4}"
        logger.info(f"🔧 DTC Match found: '{match.group(0)}' → '{result}'")
        return result
    
    # Apply the replacement
    normalized_text = re.sub(pattern, replace_dtc, text, flags=re.IGNORECASE)
    return normalized_text

@tool
def search_vehicle_documents(question: str, dtc_code: str = None, vehicle_info: str = None) -> dict:
    """Search vehicle diagnostic documents for relevant information."""
    question = normalize_dtc_codes(question)
    logger.info(f"🔍 Searching documents for: {question}") 
    

    if not retriever:
        logger.error("Retriever not available")
        return {
            "answer": "Document search not available - database not initialized.",
            "source_documents": [],
            "has_rag_info": False,
            "dtc_code": dtc_code,
            "vehicle_info": vehicle_info,
            "selected_chunk_label": "ERROR",
            "selected_chunk_content": ""
        }
    
    # Check for DTC code in question
    dtc_match = re.search(r"\b([PBUC]\d{4})\b", question.upper())
    if dtc_match:
        dtc_code = dtc_match.group(1)
    
    try:
        # Retrieve relevant documents
        docs = retriever.invoke(question)
        
        if not docs:
            return {
                "answer": "No relevant information found in the PDF.",
                "source_documents": [],
                "has_rag_info": False,
                "dtc_code": dtc_code,
                "vehicle_info": vehicle_info,
                "selected_chunk_label": "NONE",
                "selected_chunk_content": ""
            }
        
        print(f"📊 Retrieved {len(docs)} documents from vector store")

        # Display retrieved chunks with media information (like rag_test_basic)
        print("\n🔎 Retrieved Chunks:")
        for i, d in enumerate(docs, 1):
            pages = d.metadata.get("pages") or d.metadata.get("page") or "?"
            snippet = d.page_content[:300].replace('\n', ' ') + ('...' if len(d.page_content) > 300 else '')
            
            # Check for media using the same function as rag_test_basic
            media_info = extract_media_references_enhanced(d.page_content)
            media_indicators = []
            if media_info['images']:
                media_indicators.append(f"📷{len(media_info['images'])}")
            if media_info['tables']:
                media_indicators.append(f"📊{len(media_info['tables'])}")
            
            media_str = f" [{', '.join(media_indicators)}]" if media_indicators else ""
            
            print(f"  {i}. pages={pages} chars={len(d.page_content)}{media_str}")
            print(f"     snippet={snippet}")

        # Build context using the SAME format as rag_test_basic.py
        blocks = []
        for i, d in enumerate(docs, 1):
            pages = d.metadata.get("pages") or d.metadata.get("page") or "?"
            
            # Format content with media information (like rag_test_basic)
            formatted_content = format_content_with_media_enhanced(d.page_content, i)
            
            blocks.append(f"[DOC {i} | pages: {pages}]\n{formatted_content}")
        
        context = "\n\n".join(blocks)
        
        prompt = (
        "You are a helpful assistant that **must preserve all markdown elements exactly as in the context**. "
        "This includes tables, image links (`![Image](path.png)`), code blocks, and formatting.\n"
        "When answering, if images or tables are present, reproduce them exactly as they appear in the context.\n"
        "Do NOT summarize or describe them — just include them inline with the steps or content where they occur.\n"
        "If information is not available, reply exactly: I don't know.\n\n"
        f"Question: {question}\n\nContext:\n{context}\n\nAnswer (preserve markdown formatting):")

        response = llm.invoke(prompt)
        answer_text = response.content.strip()
        print(f"🤖 Direct LLM Answer: {answer_text}")
        
        if answer_text == "I don't know." or "I don't know" in answer_text:
            print(f"❌ LLM couldn't find relevant information")
            return {
                "answer": "No relevant information found in the PDF.",
                "source_documents": [],
                "has_rag_info": False,
                "dtc_code": dtc_code,
                "vehicle_info": vehicle_info,
                "selected_chunk_label": "NONE",
                "selected_chunk_content": ""
            }
        else:
            # Return the enhanced content with media information
            has_rag_info = True
            source = [{
                "page_number": docs[0].metadata.get("pages", "N/A"),
                "content": answer_text  # Return the LLM's processed answer
            }]
            
            result = {
                "answer": answer_text,
                "source_documents": source,
                "has_rag_info": has_rag_info,
                "dtc_code": dtc_code,
                "vehicle_info": vehicle_info,
                "selected_chunk_label": "PROCESSED",
                "selected_chunk_content": answer_text
            }
            return result

    except Exception as e:
        logger.error(f"Error in document search: {e}")
        return {
            "answer": "Error occurred during document search.",
            "source_documents": [],
            "has_rag_info": False,
            "dtc_code": dtc_code,
            "vehicle_info": vehicle_info,
            "selected_chunk_label": "ERROR",
            "selected_chunk_content": ""
        }

def extract_media_references_enhanced(content: str):
    """Extract image links and table references from content (same as rag_test_basic)"""
    media_info = {
        'images': [],
        'tables': [],
        'has_media': False
    }
    
    # Look for image references (common patterns)
    image_patterns = [
        r'!\[.*?\]\((.*?)\)',  # Markdown images
        r'<img.*?src=["\']([^"\']+)["\']',  # HTML images
        r'Image:\s*([^\s\n]+)',  # Custom image format
        r'Figure\s+\d+[:\-]?\s*([^\n]+)',  # Figure references
        r'\[Image:\s*([^\]]+)\]',  # Bracketed image refs
    ]
    
    for pattern in image_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        media_info['images'].extend(matches)
    
    # Look for table references
    table_patterns = [
        r'Table\s+\d+[:\-]?\s*([^\n]+)',  # Table references
        r'\|.*\|.*\|',  # Markdown table rows
        r'<table.*?</table>',  # HTML tables
    ]
    
    for pattern in table_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        media_info['tables'].extend(matches)
    
    media_info['has_media'] = bool(media_info['images'] or media_info['tables'])
    return media_info

def format_content_with_media_enhanced(content: str, doc_num: int):
    """Format content and extract media references (same as rag_test_basic)"""
    media_info = extract_media_references_enhanced(content)
    formatted_content = content
    
    # Add media information if found
    if media_info['has_media']:
        media_section = f"\n[MEDIA IN DOC {doc_num}]"
        
        if media_info['images']:
            media_section += f"\n📷 Images/Figures: {len(media_info['images'])} found"
            for i, img in enumerate(media_info['images'][:3], 1):  # Show first 3
                media_section += f"\n  - Image {i}: {img[:100]}{'...' if len(img) > 100 else ''}"
        
        if media_info['tables']:
            media_section += f"\n📊 Tables: {len(media_info['tables'])} found"
            for i, table in enumerate(media_info['tables'][:2], 1):  # Show first 2
                table_preview = table[:150].replace('\n', ' ')
                media_section += f"\n  - Table {i}: {table_preview}{'...' if len(table) > 150 else ''}"
        
        formatted_content += media_section
    
    return formatted_content

@tool
def grade_document_relevance(question: str, document_content: str, chunk_label: str = "UNKNOWN") -> dict:
    """Grade the relevance of retrieved documents (EXACT from version3_refactor)."""
    logger.info(f"📊 Grading relevance for {chunk_label}")
    print(f"🔍 GRADING DEBUG: chunk_label={chunk_label}, content_length={len(document_content if document_content else '')}")

    if not document_content or document_content == "No relevant information found in the PDF.":
        logger.info(f"📉 {chunk_label} has no content, score=0")
        result = {"relevance_score": 0, "graded": True, "chunk": chunk_label}
        print(f"🔍 GRADING RESULT: {result}")
        return result
    
    # Truncate for speed
    if len(document_content) > 300:
        truncated_content = document_content[:1000] + "..."
        logger.info(f"✂️ Truncated {chunk_label} from {len(document_content)} to 1000 chars")
    else:
        truncated_content = document_content
    
    try:
        score = retrieval_grader.invoke({"question": question, "documents": truncated_content})
        relevance_score = score.get('score', 0)
        logger.info(f"✅ {chunk_label} relevance score = {relevance_score}")
        result = {
            "relevance_score": relevance_score,
            "graded": True,
            "chunk": chunk_label
        }
        print(f"🔍 GRADING RESULT: {result}")
        return result
    except Exception as e:
        logger.error(f"⚠️ Error grading {chunk_label}: {e}")
        result = {"relevance_score": 0, "graded": True, "chunk": chunk_label}
        print(f"🔍 GRADING RESULT (error fallback): {result}")
        return result

@tool
def search_web_for_vehicle_info(query: str, dtc_code: str = None, vehicle_info: str = None) -> dict:
    """Search web for additional vehicle diagnostic information using Tavily (EXACT from version3_refactor)."""
    logger.info("🌐 Performing Tavily web search")
    
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.warning("No Tavily API key found, skipping web search")
        return {"results": [], "success": False, "error": "Missing TAVILY_API_KEY"}
    
    try:
        from tavily import TavilyClient
        tavily_client = TavilyClient(api_key=tavily_api_key)
    except ImportError:
        logger.warning("Tavily package not installed, skipping web search")
        return {"results": [], "success": False, "error": "Tavily package not available"}
    
    search_term = dtc_code or vehicle_info or query
    
    if dtc_code:
        search_query = f"{search_term} vehicle diagnostic trouble code causes solutions"
        logger.info(f"Searching for DTC code: {search_term}")
    elif vehicle_info:
        search_query = f"{search_term} common problems solutions"
        logger.info(f"Searching for vehicle info: {search_term}")
    else:
        search_query = search_term
        logger.info(f"Searching for general query: {search_term}")
    
    try:
        response = tavily_client.search(query=search_query, max_results=3, search_depth="advanced")
        results = response.get('results', [])
        logger.info(f"Found {len(results)} results from web search")
        
        return {
            "results": results,
            "success": True,
            "query_used": search_query
        }
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return {"results": [], "success": False, "error": str(e)}

@tool
def search_youtube_videos(query: str, dtc_code: str = None, vehicle_info: str = None) -> dict:
    """Search YouTube for diagnostic videos (EXACT from version3_refactor)."""
    logger.info("📺 Performing YouTube search")
    
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    if not YOUTUBE_API_KEY:
        logger.warning("YouTube API key not found in environment variables")
        return {"youtube_results": [], "success": False,"error": "Missing YOUTUBE_API_KEY"}
    
    search_term = dtc_code or vehicle_info or query
    
    if dtc_code:
        search_query = f"{search_term} diagnostic trouble code repair"
        logger.info(f"Searching YouTube for DTC code: {search_term}")
    elif vehicle_info:
        search_query = f"{search_term} repair maintenance"
        logger.info(f"Searching YouTube for vehicle: {search_term}")
    else:
        search_query = f"car {search_term}"
        logger.info(f"Searching YouTube for general query: {search_term}")
    
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": search_query,
        "key": YOUTUBE_API_KEY,
        "maxResults": 4,
        "type": "video"
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            logger.error(f"YouTube API error: {response.status_code} {response.text}")
            return {"youtube_results": [], "success": False}
        
        data = response.json()
        videos = []
        
        for item in data.get("items", []):
            video_id = item["id"]["videoId"]
            videos.append({
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "video_id": video_id,
                "title": item["snippet"]["title"],
                "thumbnail_hq": item["snippet"]["thumbnails"].get("high", {}).get("url", ""),
                "thumbnail_max": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            })
        
        logger.info(f"Found {len(videos)} YouTube videos")
        return {
            "youtube_results": videos,
            "success": True,
            "query_used": search_query
        }
    except Exception as e:
        logger.error(f"YouTube search error: {e}")
        return {"youtube_results": [], "success": False, "error": str(e)}

@tool
def format_diagnostic_results(
    question: str,
    document_content: str,
    web_results: Optional[List[dict]] = None,
    youtube_results: Optional[List[dict]] = None,
    dtc_code: Optional[str] = None,
    relevance_score: int = 0,
) -> dict:
    """Return structured dual output (voice + text) per new requirements.

    Output JSON string inside formatted_response:
    {
      "voice_output": str,
      "text_output": {
          "content": str,
          "web_sources": [ {"title": str, "url": str}, ... ],
          "youtube_videos": [ {"title": str, "url": str, "video_id": str, "thumbnail": str}, ... ]
      }
    }
    """
    logger.info("📝 Formatting final results (structured)")
    rag_answer = document_content

    # Decode potential JSON from prior tools
    if isinstance(rag_answer, str):
        try:
            rag_data = json.loads(rag_answer)
            rag_content = rag_data.get('answer', rag_answer)
            print(f"✅ Parsed JSON answer length: {len(rag_content)}")
        except json.JSONDecodeError:
            rag_content = rag_answer
            print(f"✅ Raw answer length: {len(rag_content)}")
    else:
        rag_content = str(rag_answer)

    has_rag_info = (
        rag_content
        and rag_content != "No relevant information found in the PDF."
        and "No relevant information found" not in rag_content
    )
    use_rag = has_rag_info and relevance_score == 1

    # --- RAG PATH ---
    if use_rag:
        logger.info("Using RAG content for detailed text output")
        processed_rag_content = process_content_with_inline_images_enhanced(rag_content)

        # Voice summary generation (strip markdown/images/links)
        plain_voice = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", processed_rag_content)
        plain_voice = re.sub(r"\[[^\]]*\]\([^)]*\)", "", plain_voice)
        plain_voice = re.sub(r"[*`>#_]+", "", plain_voice)
        voice_prompt = (
            "Summarize the diagnostic information in 30-35 words, natural speech, no URLs, no asterisks."\
            " Avoid reading image references.\nCONTENT:\n" + plain_voice[:1500]
        )
        try:
            voice_output = llm.invoke(voice_prompt).content.strip()
            voice_output = re.sub(r"\s+", " ", voice_output)
        except Exception as e:
            logger.warning(f"Voice summary LLM error (RAG path): {e}")
            voice_output = "Here is a concise diagnostic summary from retrieved documentation."

        structured = {
            "voice_output": voice_output,
            "text_output": {
                "content": processed_rag_content,
                "web_sources": [],
                "youtube_videos": [],
            },
        }
        return {"formatted_response": json.dumps(structured, ensure_ascii=False)}

    # --- WEB/YOUTUBE PATH ---
    if not use_rag and (web_results or youtube_results):
        logger.info("Using web + YouTube results (RAG insufficient or low relevance)")
        # Normalize web sources
        web_sources_struct: List[Dict[str, str]] = []
        for r in web_results or []:
            url = r.get("url", "")
            if not url:
                continue
            title = r.get("title") or url
            web_sources_struct.append({"title": title, "url": url})

        # Normalize YouTube videos
        youtube_struct: List[Dict[str, str]] = []
        for v in youtube_results or []:
            video_id = v.get("video_id", "")
            thumb = v.get("thumbnail_hq") or v.get("thumbnail_max") or (
                video_id and f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
            ) or ""
            youtube_struct.append({
                "title": v.get("title", "Video"),
                "url": v.get("url", ""),
                "video_id": video_id,
                "thumbnail": thumb,
            })

        web_content_text = "\n\n".join([
            r.get("content", "") for r in web_results or [] if r.get("content")
        ])

        if dtc_code:
            synthesis_prompt = f"""You are an automotive diagnostic technician.
Create structured diagnostic details for DTC {dtc_code}.
Sections with bullet points starting with '•':
Category:
Potential Causes:
Diagnostic Steps:
Possible Solutions:
Avoid URLs or YouTube links in bullet body.
Source text:
{web_content_text[:3000]}
"""
        else:
            synthesis_prompt = f"""You are an automotive assistant.
Provide structured diagnostic details for QUESTION: {question}
Use bullet points with '•'. Avoid URLs inside bullets.
Source text:
{web_content_text[:3000]}
"""
        try:
            synthesized = llm.invoke(synthesis_prompt).content.strip()
        except Exception as e:
            logger.warning(f"Synthesis LLM error: {e}")
            synthesized = "Diagnostic information compiled from external sources.";

        voice_prompt = (
            "Give a spoken summary in <= 30 words of the diagnostic findings. No lists, no URLs."\
            "\nCONTENT:\n" + synthesized[:1200]
        )
        try:
            voice_output = llm.invoke(voice_prompt).content.strip()
            voice_output = re.sub(r"\s+", " ", voice_output)
        except Exception as e:
            logger.warning(f"Voice summary LLM error (web path): {e}")
            voice_output = "Brief diagnostic summary from web and video sources.";

        structured = {
            "voice_output": voice_output,
            "text_output": {
                "content": synthesized,
                "web_sources": web_sources_struct,
                "youtube_videos": youtube_struct,
            },
        }
        return {"formatted_response": json.dumps(structured, ensure_ascii=False)}

    # --- FALLBACK (low relevance, no web results) ---
    logger.info("Fallback path: RAG low relevance, no web results")
    processed = process_content_with_inline_images_enhanced(rag_content)
    plain_voice = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", processed)
    plain_voice = re.sub(r"\[[^\]]*\]\([^)]*\)", "", plain_voice)
    plain_voice = re.sub(r"[*`>#_]+", "", plain_voice)
    fallback_voice = plain_voice.split('\n')[0][:140] or "Diagnostic information unavailable.";
    structured = {
        "voice_output": fallback_voice,
        "text_output": {
            "content": processed,
            "web_sources": [],
            "youtube_videos": [],
        },
    }
    return {"formatted_response": json.dumps(structured, ensure_ascii=False)}

def process_content_with_inline_images_enhanced(content: str) -> str:
    """Enhanced processing to preserve images and improve step formatting."""
    
    if not content or len(content) < 10:
        return content
    
    try:
        lines = content.split('\n')
        processed_lines = []
        
        for line in lines:
            original_line = line
            line = line.strip()
            
            # Skip empty lines but preserve them for spacing
            if not line:
                processed_lines.append("")
                continue
            
            # ✅ CRITICAL: Preserve image references EXACTLY as they are
            if '![Image](' in line:
                processed_lines.append(f"\n{line}\n")  # Add spacing around images
                print(f"🖼️ Preserved image: {line[:50]}...")
                continue
            
            # Check for section headers (## DIAGNOSIS, etc.)
            if line.startswith('##'):
                processed_lines.append(f"\n**{line[2:].strip().upper()}**\n")
                continue
            
            # Check for numbered steps with better formatting
            step_match = re.match(r'^(\d+)\.\s*(.*)', line)
            if step_match:
                step_num = step_match.group(1)
                step_text = step_match.group(2)
                processed_lines.append(f"\n**Step {step_num}: {step_text}**\n")
                continue
            
            # Check for YES/NO decision points and format them prominently
            if line.startswith(('YES -', 'NO -', 'YES:', 'NO:')):
                processed_lines.append(f"\n   ✅ **{line}**")
                continue
            
            # Check for bullet points and sub-steps
            elif line.startswith(('-', '•')):
                processed_lines.append(f"   {line}")
                continue
            
            # Check for table rows (contains | characters) - preserve exactly
            elif '|' in line and len([c for c in line if c == '|']) >= 2:
                processed_lines.append(line)  # Keep table formatting as-is
                continue
            
            # Check for table separators
            elif '---' in line and '|' in line:
                processed_lines.append(line)
                continue
            
            # Check for questions (ending with ?)
            elif line.endswith('?'):
                processed_lines.append(f"\n**{line}**")
                continue
            
            # Check for "PART NUMBER" labels and similar
            elif line.upper() in ['PART NUMBER', 'REPAIR PROCEDURE', 'DIAGNOSIS']:
                processed_lines.append(f"\n**{line.upper()}**")
                continue
            
            # Regular text - preserve as-is but check for special formatting
            else:
                if line:
                    # Check if it's a procedure note or important info
                    if any(keyword in line.lower() for keyword in ['check', 'remove', 'install', 'replace', 'update']):
                        processed_lines.append(f"   • {line}")
                    else:
                        processed_lines.append(line)
        
        result = '\n'.join(processed_lines)
        
        # Final cleanup - ensure proper spacing around images
        result = re.sub(r'\n{3,}', '\n\n', result)  # Max 2 consecutive newlines
        
        print(f"🎨 Enhanced processing complete:")
        print(f"   - Original length: {len(content)}")
        print(f"   - Processed length: {len(result)}")
        print(f"   - Images preserved: {result.count('![Image](')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in process_content_with_inline_images_enhanced: {e}")
        return content

def process_content_with_inline_images(content: str) -> str:
    """Process content to display images and tables inline with steps (EXACT from version3_refactor)."""
    
    if not content or len(content) < 10:
        return content
    
    try:
        lines = content.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Check for numbered steps
            if re.match(r'^\d+\.\s+', line):
                processed_lines.append(f"**{line}**")
            
            # Check for YES/NO decision points and bullet points
            elif line.startswith(('YES -', 'NO -', '- YES', '- NO', '-')):
                processed_lines.append(f"   **{line}**")
            
            # Check for image references - show markdown as-is
            elif '![Image](' in line:
                processed_lines.append(f"   {line}")
                processed_lines.append("")  # Space after image
            
            # Check for table rows (contains | characters)
            elif '|' in line and len([c for c in line if c == '|']) >= 2:
                processed_lines.append(line)  # Keep table formatting as-is
            
            # Check for table separators
            elif '---' in line and '|' in line:
                processed_lines.append(line)
            
            # Check for questions (ending with ?)
            elif line.endswith('?'):
                processed_lines.append(f"\n**{line}**")
            
            else:
                if line:
                    processed_lines.append(line)
        
        return '\n'.join(processed_lines)
        
    except Exception as e:
        logger.error(f"Error in process_content_with_inline_images: {e}")
        return content

# Export all tools
__all__ = [
    "is_vehicle_related",
    "extract_vehicle_model", 
    "search_vehicle_documents",
    "grade_document_relevance",
    "search_web_for_vehicle_info",
    "search_youtube_videos",
    "format_diagnostic_results",
    "llm",
    "retriever",
]
