// Groq via OpenAI-compatible API (same endpoint as src/agents/clients/groq_client.py).

client<llm> GroqLlama {
  provider openai-generic
  options {
    base_url "https://api.groq.com/openai/v1"
    api_key env.GROQ_API_KEY
    model "llama-3.3-70b-versatile"
    temperature 0.2
  }
}
