# Voice and Text Output Differentiation - Implementation Summary

## Overview
This implementation separates voice output (for TTS) and text output (for chat display) to provide:
- **Voice Output**: Concise summaries suitable for text-to-speech
- **Text Output**: Complete detailed information with URLs, images, and structured data

## Backend Changes Made

### 1. Modified `tools/RAG_tools.py`

#### Enhanced `format_diagnostic_results` function:
- Returns structured JSON containing both voice and text outputs
- Voice output: Concise summary without URLs/images
- Text output: Complete information with web sources and YouTube videos

#### Added `_create_voice_summary` function:
- Creates voice-friendly summaries by removing markdown, URLs, and formatting
- Uses LLM to generate conversational summaries under 150 words
- Handles both DTC codes and general automotive queries

#### Structured Output Format:
```json
{
  "formatted_response": "Complete response (backward compatibility)",
  "voice_output": "Concise voice summary",
  "text_output": {
    "content": "Detailed content",
    "web_sources": [{"url": "...", "title": "..."}],
    "youtube_videos": [{"url": "...", "title": "...", "thumbnail": "...", "video_id": "..."}],
    "has_external_sources": true
  },
  "source_type": "web_search" | "rag" | "rag_fallback" | "fallback"
}
```

### 2. Modified `workflows/diagnostic_workflow.py`

#### Enhanced `_get_response` method:
- Detects structured JSON responses from tools
- Formats text output with clickable web sources and YouTube videos
- Maintains backward compatibility with plain text responses

#### Updated `_process_message` method:
- Handles structured responses from `format_diagnostic_results`
- Passes JSON responses through for LiveKit processing

## Frontend Changes Required

### 1. Enhanced Message Formatting (`components/livekit/chat/hooks/utils.ts`)
- Added `enhancedMessageFormatter` to handle:
  - Markdown-style links: `[title](url)` → clickable links
  - Plain URLs → clickable links
  - Bold text formatting: `**text**` → `<strong>text</strong>`
  - Section headers for Web Sources and YouTube videos

### 2. Structured Message Rendering
The backend now returns formatted text with:
- **Web Sources**: Clickable links with titles
- **YouTube Videos**: Formatted as links (thumbnails would need additional frontend component)

## Key Features Implemented

### Voice Output Differentiation
- ✅ Concise voice summaries (no URLs, images, or technical formatting)
- ✅ LLM-generated conversational summaries
- ✅ Under 150 words for optimal TTS experience

### Text Output Enhancement
- ✅ Complete detailed information
- ✅ Clickable web source links with titles
- ✅ YouTube video links with metadata
- ✅ Structured data for web search results
- ✅ Backward compatibility with RAG-only results

### Response Routing Logic
- ✅ RAG database results (relevance_score = 1) → Use RAG content
- ✅ No RAG match (relevance_score = 0) → Use web search + YouTube
- ✅ Fallback handling for error cases

## Testing

### Test Files Created
1. `test_structured_output.py` - Full workflow test
2. `test_simple_rag.py` - Tool-level test

### Expected Behavior
1. **Voice Query**: User asks about P0171 DTC code
2. **RAG Check**: System checks diagnostic database
3. **Web Search**: If no match, searches web + YouTube
4. **Dual Output**:
   - Voice: "P0171 indicates a lean fuel mixture in bank 1. Main causes include vacuum leaks and faulty oxygen sensors. Check for vacuum leaks, test oxygen sensors, and inspect the air filter."
   - Text: Complete formatted response with:
     - Detailed diagnostic information
     - Web sources with clickable links
     - YouTube videos with titles and links

## Frontend Rendering Improvements

### Before
- Plain text responses
- No clickable links
- No media integration

### After
- Formatted responses with sections
- Clickable web source links
- YouTube video links (ready for thumbnail integration)
- Bold headings and structured layout

## Additional Considerations

### Future Enhancements
1. **YouTube Thumbnails**: Add thumbnail display component
2. **Image Rendering**: Handle diagnostic images from web sources
3. **Voice Control**: Allow users to request "voice summary" vs "full details"
4. **Caching**: Cache voice summaries for repeated queries

### Performance
- Voice summaries are generated on-demand
- Structured data is efficiently formatted
- Backward compatibility maintained for existing queries

## Configuration
The system uses environment variables:
- `TAVILY_API_KEY` - For web search
- `YOUTUBE_API_KEY` - For YouTube video search  
- `OPENAI_API_KEY` - For LLM processing and voice summary generation