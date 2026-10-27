import streamlit as st
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from operator import itemgetter
# from dotenv import load_dotenv
# load_dotenv(r'D:\Learning\TuteDude2026\Python\GenAI_Lectures_Apps\.env')

st.sidebar.title("ChatBot With History and Memory Trimmeing")
groq_api_key = st.sidebar.text_input("Provide your Groq API Key", type='password')

if 'chat_history' not in st.session_state or st.sidebar.button('Clear Chat History'):
    st.session_state.chat_history = [
        AIMessage(content="Hello! How can I help you today?")
    ]

# 3. Render previous chat messages
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

prompt = ChatPromptTemplate.from_messages([
    ('system',"""You are helpful AI Assitant, please look for answer in  
    chat hostory provided first and if you dont find the answer in chat history
    then use your own knowledge"""),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

if groq_api_key:
    model = ChatGroq(model="openai/gpt-oss-20b",temperature=0.5,api_key=groq_api_key)

    trimmer = trim_messages(max_tokens=1500,allow_partial=False,token_counter=model,start_on='human')

    chain = (RunnablePassthrough.assign(chat_history=itemgetter('chat_history')|trimmer)
    | prompt | model | StrOutputParser())

    def generate_response(query):
        result = chain.invoke({"query":query,'chat_history':st.session_state.chat_history})
        st.session_state.chat_history.append(HumanMessage(content=query))
        st.session_state.chat_history.append(AIMessage(content=result))
        return result

    query = st.text_input("Ask Anything !!")
    st.chat_message('user').write(query)

    if st.button("Answer"):
        if query:
            response = generate_response(query)
            st.write(response)
        else:
            st.warning("Please enter you query")
else:
    st.info("Provide Groq API Key.")