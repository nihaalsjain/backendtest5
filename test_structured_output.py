#!/usr/bin/env python3
"""
Test script to verify the structured output format from the modified RAG tools.
"""

import sys
import os
import asyncio
import json

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from workflows.diagnostic_workflow import AsyncDiagnosticAgent

async def test_structured_output():
    """Test the structured output format."""
    print("🧪 Testing structured output format...")
    
    try:
        # Create diagnostic agent
        agent = AsyncDiagnosticAgent(target_language="en")
        await agent.initialize()
        
        # Test with a simple automotive query
        test_query = "What does diagnostic trouble code P0171 mean?"
        print(f"🔍 Test query: {test_query}")
        
        # Get response
        response = await agent.chat(test_query)
        print(f"📋 Raw response: {response[:200]}...")
        
        # Check if response is structured JSON
        try:
            if response.strip().startswith('{'):
                structured_data = json.loads(response)
                print("✅ Response is structured JSON!")
                print(f"📝 Keys: {list(structured_data.keys())}")
                
                if "voice_output" in structured_data:
                    print(f"🗣️ Voice output: {structured_data['voice_output'][:100]}...")
                
                if "text_output" in structured_data:
                    text_out = structured_data['text_output']
                    if isinstance(text_out, dict):
                        print(f"📄 Text output (structured): {text_out.get('content', '')[:100]}...")
                        if 'web_sources' in text_out:
                            print(f"🌐 Web sources: {len(text_out['web_sources'])} found")
                        if 'youtube_videos' in text_out:
                            print(f"📺 YouTube videos: {len(text_out['youtube_videos'])} found")
                    else:
                        print(f"📄 Text output (simple): {str(text_out)[:100]}...")
            else:
                print("ℹ️ Response is plain text (not structured)")
                
        except json.JSONDecodeError:
            print("⚠️ Response is not valid JSON")
        
        # Cleanup
        await agent.cleanup()
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_structured_output())