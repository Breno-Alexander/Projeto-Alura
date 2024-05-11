# Projeto-Alura
from google.generativeai import genai
import pinecone

# Configurar API Gemini e Pinecone (banco de dados vetorial)
genai.configure(api_key="SUA_CHAVE_API_GEMINI")
pinecone.init(api_key="SUA_CHAVE_API_PINECONE", environment="SEU_AMBIENTE_PINECONE")

# Criar índice no Pinecone
index = pinecone.Index("seu_indice")

# Função para gerar embeddings com Gemini
def gerar_embedding(texto):
  response = genai.generate_embeddings(model="gemini", texts=[texto])
  return response.embeddings[0]

# Função para indexar documentos
def indexar_documento(documento):
  embedding = gerar_embedding(documento["conteudo"])
  metadados = {"titulo": documento["titulo"], "url": documento["url"]}
  index.upsert([(documento["id"], embedding, metadados)])

# Função para buscar documentos
def buscar_documentos(consulta):
  consulta_embedding = gerar_embedding(consulta)
  resultados = index.query(vector=consulta_embedding, top_k=10, include_metadata=True)
  return resultados

# Exemplo de uso
documentos = [
  {"id": 1, "titulo": "Documento 1", "url": "https://exemplo.com/doc1", "conteudo": "Este é o conteúdo do documento 1."},
  {"id": 2, "titulo": "Documento 2", "url": "https://exemplo.com/doc2", "conteudo": "Este é o conteúdo do documento 2."},
]

for documento in documentos:
  indexar_documento(documento)

consulta = "informações sobre documento 1"
resultados = buscar_documentos(consulta)

for resultado in resultados["matches"]:
  print(f"Título: {resultado['metadata']['titulo']}")
  print(f"URL: {resultado['metadata']['url']}")
  print(f"Similaridade: {resultado['score']}\n")
