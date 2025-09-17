from fastapi import FastAPI, HTTPException, Body, Header
from fastapi.middleware.cors import CORSMiddleware
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import json
import traceback

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-05-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

ASSISTANT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
TOKEN_ID = os.getenv("token_id")


def validar_token(token_id_header: str = Header(...)):
    if token_id_header != TOKEN_ID:
        raise HTTPException(status_code=401, detail="Token inválido")


def enviar_para_assistente(deployment_name: str, texto: str):
    if not deployment_name:
        raise Exception("Deployment name não fornecido")

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "Você é um revisor de deliberações e deve responder em JSON."},
            {"role": "user", "content": texto}
        ]
    )

    ultima_resposta = response.choices[0].message.content

    if isinstance(ultima_resposta, str):
        try:
            resposta_tratada = ultima_resposta.strip()
            if (resposta_tratada.startswith('"') and resposta_tratada.endswith('"')) or \
               (resposta_tratada.startswith("'") and resposta_tratada.endswith("'")):
                resposta_tratada = resposta_tratada[1:-1]
            if resposta_tratada.startswith('```json'):
                resposta_tratada = resposta_tratada[7:]
            if resposta_tratada.startswith('```'):
                resposta_tratada = resposta_tratada[3:]
            if resposta_tratada.endswith('```'):
                resposta_tratada = resposta_tratada[:-3]
            resposta_tratada = resposta_tratada.strip()
            try:
                return json.loads(resposta_tratada)
            except Exception as e:
                raise Exception(f"Resposta do assistente não é JSON válido. Resposta recebida: {repr(resposta_tratada)}")
        except json.JSONDecodeError:
            raise Exception("Resposta do assistente não é JSON válido")
    return ultima_resposta


@app.get("/version")
def version():
    return {"version": "1.0.0"}


@app.post("/revisor")
def revisor_deliberacao(
    payload: dict = Body(...),
    token_id: str = Header(...),
    assistant_id: str | None = Header(None)
):
    try:
        validar_token(token_id)

        texto = payload.get("texto")
        if not texto:
            raise HTTPException(status_code=400, detail="Campo 'texto' é obrigatório")

        deployment = assistant_id if assistant_id else ASSISTANT_DEPLOYMENT
        if not deployment:
            raise HTTPException(status_code=500, detail="Deployment não configurado.")

        resposta = enviar_para_assistente(deployment, texto)
        return resposta
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


print("Deployment padrão (revisor):", ASSISTANT_DEPLOYMENT)
print("Token configurado:", "Sim" if TOKEN_ID else "Não")
