from langchain_community.llms import Ollama
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


query = "可以跟我说5个关于龙舟的趣事吗？"

llm = Ollama(model="llama2-chinese",
             callbacks=CallbackManager([StreamingStdOutCallbackHandler()]))
llm.invoke(query)
