from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline
import textwrap

loader = PyPDFLoader(r"path/to/the/pdf")
docs = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(docs)

summarizer_pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=200,
    truncation=True,
    temperature=0.3,
    device=-1   
)
llm = HuggingFacePipeline(pipeline=summarizer_pipe)

prompt = PromptTemplate.from_template("""
Summarize the following text in 3-5 concise bullet points:
{text}
""")

chain = LLMChain(llm=llm, prompt=prompt)

summaries = []
for i, chunk in enumerate(chunks): 
    print(f"\nSummarizing chunk {i+1}/{len(chunks)} ...")
    summary = chain.run(chunk.page_content)
    summaries.append(summary.strip())

final_summary_input = "\n".join(summaries)
final_summary_prompt = PromptTemplate.from_template("""
Combined summary:
{input}
""")
combine_chain = LLMChain(llm=llm, prompt=final_summary_prompt)

final_summary = combine_chain.run({"input": final_summary_input})

print("\nSUMMARY:\n")
print(textwrap.fill(final_summary, width=100))
