#!/usr/bin/env python3
"""
Simplified test to verify the RAG tools return structured output.
"""

import json
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(__file__))

try:
    from tools.RAG_tools import format_diagnostic_results
    print("✅ Successfully imported format_diagnostic_results")
    
    # Test the tool with sample data
    test_web_results = [
        {
            "url": "https://example.com/dtc-p0171",
            "title": "Understanding DTC P0171",
            "content": "DTC P0171 indicates a lean fuel mixture in bank 1."
        }
    ]
    
    test_youtube_results = [
        {
            "url": "https://youtube.com/watch?v=test123",
            "title": "How to Fix P0171 Error Code",
            "thumbnail_hq": "https://img.youtube.com/vi/test123/hqdefault.jpg",
            "video_id": "test123"
        }
    ]
    
    # Call the tool
    result = format_diagnostic_results(
        question="What does P0171 mean?",
        rag_answer="No relevant information found in the PDF.",
        web_results=test_web_results,
        youtube_results=test_youtube_results,
        dtc_code="P0171",
        relevance_score=0  # Low score to trigger web search
    )
    
    print(f"📄 Tool result type: {type(result)}")
    print(f"📄 Result preview: {str(result)[:200]}...")
    
    # Try to parse as JSON
    try:
        parsed_result = json.loads(result)
        print("✅ Result is valid JSON!")
        print(f"🔑 Keys: {list(parsed_result.keys())}")
        
        if "voice_output" in parsed_result:
            print(f"🗣️ Voice output available: {len(parsed_result['voice_output'])} chars")
            
        if "text_output" in parsed_result:
            text_out = parsed_result["text_output"]
            if isinstance(text_out, dict):
                print(f"📄 Structured text output with keys: {list(text_out.keys())}")
            else:
                print(f"📄 Simple text output: {len(str(text_out))} chars")
                
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()