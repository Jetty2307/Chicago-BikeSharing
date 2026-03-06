import os
import requests
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

client_OpenAI = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
)


tools = [{
    "type": "function",
    "function": {
        "name": "make_summary",
        "description": "Returns the description of the model's performance",
        "parameters": {"model" : "the name of the model you used for description generation (e.g., 'deepseek-chat')",
                       "model_params": "parameters and metrics of the model (e.g., 'model_params')"}
    }
}]

def performance_summary(model, model_params: dict):
    """Generates a concise summary of the model's applicability in time series analysis,
    highlighting key strengths and weaknesses based on the provided parameters and metrics."""

    prompt = f"""You have trained a model with the following parameters and metrics: {model_params}.
    Please provide a concise summary of the model's applicability in time series analysis, model's performance, 
    highlighting key strengths and weaknesses 
    based on the provided parameters and metrics. Focus on the most relevant aspects that would help in 
    understanding how well the model is performing and where it may need improvement.
    Return the summary as a single paragraph, without any bullet points or lists. 
    """.strip()

    # ollama.chat

    if model == "llama3":
        try:
            ollama_response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "llama3",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=(5, 300),
            )
            ollama_response.raise_for_status()
            description = ollama_response.json()["message"]["content"].strip()
            return f"{description}\n Provided by {model}"
        except requests.exceptions.RequestException:
            return performance_summary("deepseek-chat", model_params)

    try:
        response = client_OpenAI.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        description = response.choices[0].message.content.strip()
    except TypeError as exc:
        if "by_alias" not in str(exc):
            raise

        # Fallback for SDK/pydantic incompatibility in some local environments.
        http_response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=60,
        )
        http_response.raise_for_status()
        description = http_response.json()["choices"][0]["message"]["content"].strip()

    return f"{description}\n Provided by {model}"


def reflect_on_description(description: str, model_params: dict, model: str = "deepseek-chat") -> str:
    messages = [
        {"role": "system", "content": f""""You are ML expert. Please decide on the quality of the description. 
        If the description is satisfactory return ONLY 'OK', if the description is bad return ONLY 'NOT'"""},
        {"role": "user", "content": f"MODEL_PARAMS_AND_METRICS: {model_params}\n\nSUMMARY:\n{description}"}
    ]

    try:
        response = client_OpenAI.chat.completions.create(
            model=model,
            messages=messages,
            stream=False
        )
        reflection = response.choices[0].message.content.strip()
    except TypeError as exc:
        if "by_alias" not in str(exc):
            raise

        http_response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={"model": model, "messages": messages, "stream": False},
            timeout=60,
        )
        http_response.raise_for_status()
        reflection = http_response.json()["choices"][0]["message"]["content"].strip()

    return reflection.upper()


def revise_description(reflection: str,
                       original_description: str,
                       model_params: dict):
    print(reflection)
    if reflection == "OK":
        return original_description
    elif reflection == "NOT":
        return performance_summary("deepseek-chat", model_params)
        # response = client_OpenAI.chat.completions.create(
        #     model="deepseek-chat",
        #     messages=[
        #         {"role": "user", "content": f""""Generate a description with {model_params}"""},
        #     ],
        #     tools=tools,
        #     max_turns=2,
        #     stream=False
        # )
        # return response.choices[0].message.content

    else:
        return None

def make_summary(model_params):
    description = performance_summary('llama3', model_params)
    reflection = reflect_on_description(description, model_params)
    return revise_description(reflection, description, model_params)

