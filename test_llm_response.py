# test_llm_response.py
import requests
import json

def test_llm_response():
    try:
        # Test a simple prompt with the same settings we're using
        data = {
            "model": "llama3.1",
            "prompt": "You are an expert SRE. What is the root cause of a NullPointerException in Java?",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 100
            }
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=data, timeout=15)
        if response.status_code == 200:
            result = response.json()
            print("✅ LLM response test successful")
            print(f"Response: {result['response']}")
            return True
        else:
            print(f"❌ LLM response test failed with status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing LLM: {e}")
        return False

if __name__ == "__main__":
    test_llm_response()