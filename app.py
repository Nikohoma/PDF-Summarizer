from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from transformers import pipeline



loader = PyPDFLoader(r"path/to/the/pdf") 
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(docs)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

summarizer_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=300)
llm = HuggingFacePipeline(pipeline=summarizer_pipe)





prompt = ChatPromptTemplate.from_template("""
Summarize the following context in clear and concise bullet points.
Context:
{context}

Question:
{question}
""")


rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
response = rag_chain.invoke("")

print("Summary:\n")
print(response)
