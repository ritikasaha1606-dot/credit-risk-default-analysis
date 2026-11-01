import time
import streamlit as st
from google import genai
from google.genai import types
from tools import default_rate_by, score_borrower, top_risk_factors

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
MODEL = "gemini-3.8-flash"
TOOLS = {'default_rate_by': default_rate_by,
         'score_borrower': score_borrower,
         'top_risk_factors': top_risk_factors}

SYSTEM_PROMPT = """..."""   # the prompt from step 6

def call_model(contents, config, tries=4):
    for i in range(tries):
        try:
            return client.models.generate_content(
                model=MODEL, contents=contents, config=config)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)

def run_agent(question, max_rounds=3):
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=list(TOOLS.values()),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]
    log = []

    for _ in range(max_rounds):
        resp = client.models.generate_content(
            model=MODEL, contents=contents, config=config)

        if not resp.function_calls:          # no tool needed: final answer
            return resp.text, log

        contents.append(resp.candidates[0].content)   # keep the model's request
        parts = []
        for call in resp.function_calls:
            args = dict(call.args)
            if call.name in TOOLS:
                try:
                    result = TOOLS[call.name](**args)
                except Exception as e:
                    result = {"error": str(e)}
            else:
                result = {"error": f"Unknown tool {call.name}"}
            log.append((call.name, args))
            parts.append(types.Part.from_function_response(
                name=call.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=parts))

    return "Stopped after max rounds without a final answer.", log