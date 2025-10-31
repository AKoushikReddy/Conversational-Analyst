# Conversational Analyst (Dev Setup)

## Local Development

```bash
# 1️⃣ Start Ollama in a separate terminal
ollama serve

# 2️⃣ Run the project (this script sets up env + starts FastAPI)
./run_dev.sh
```

## Notes
- Ensure `ollama` CLI is installed and reachable via http://localhost:11434.
- Adjust the model in the script if you're using something other than `llama3`.
- FastAPI will run at http://localhost:8000.
