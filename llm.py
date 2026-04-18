from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_classic import hub
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader

def get_ai_message(user_message):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    loader = Docx2txtLoader('./tax-markdown.docx')
    document_list = loader.load_and_split(text_splitter)

    # Existing Pinecone index dimension is 1024.
    embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=1024)
    index_name = 'tax-markdown-index'
    database = PineconeVectorStore.from_documents(embedding=embedding, index_name=index_name, documents=document_list)

    llm = ChatOpenAI(model='gpt-4o')
    prompt = hub.pull("rlm/rag-prompt")

    retriever = database.as_retriever(search_kwargs={'K': 4})

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever,
                                        chain_type_kwargs={"prompt": prompt},
                                        )

    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    prompt = ChatPromptTemplate.from_template(f"""
            사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
            만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
            그런 경우에는 질문만 리턴해주세요.
            사전: {dictionary}
            
            질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    tax_chain = {"query": dictionary_chain} | qa_chain
    ai_message = tax_chain.invoke({"question": user_message})
    
    return ai_message["result"]