from langchain_community.llms import Ollama


query = "可以跟我说5个关于龙舟的趣事吗？"

llm = Ollama(model="llama2-chinese")

for chunks in llm.stream(query):
    print(chunks, end='', sep='')
