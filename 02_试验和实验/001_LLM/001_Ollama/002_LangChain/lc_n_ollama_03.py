from langchain_community.llms import Ollama
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate


llm = Ollama(model="llama2-chinese",
             callbacks=CallbackManager([StreamingStdOutCallbackHandler()]),
             temperature=0.9)

prompt = PromptTemplate.from_template("可以跟我说5个关于{something}的趣事吗？")

chain = prompt | llm

print(chain.invoke("龙舟"))
