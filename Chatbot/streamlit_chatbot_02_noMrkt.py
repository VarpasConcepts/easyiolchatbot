import os
import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI as LangChainOpenAI
import time

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI client
client = OpenAI()

# Directory where the FAISS index is stored
storage_directory = "faiss_index"

@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    try:
        index_path = os.path.join(os.path.dirname(__file__), storage_directory)
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading index: {e}")
        st.error("The FAISS index file is missing or cannot be accessed. Please check the file path and permissions.")
        return None

def query_knowledge_base(query, vectorstore):
    if vectorstore is None:
        return None, None

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = LangChainOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    try:
        result = qa_chain({"query": query})
        return result["result"], result["source_documents"]
    except Exception as e:
        st.error(f"Error querying knowledge base: {e}")
        return None, None

def chat_with_gpt(messages):
    try:
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant educating patients about IOLs. Please aim to keep your responses concise, ideally between 150-200 words. Cover all relevant information but prioritize brevity and clarity. If a longer response is absolutely necessary to adequately address the query, you may exceed this limit, but strive to be as concise as possible."
        }
        
        messages.insert(0, system_message)
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.7
        )
        time.sleep(1)
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating ChatGPT response: {e}")
        return None

def process_query(query, vectorstore, user_lifestyle, prioritized_lenses):
    if not st.session_state.show_lens_options:
        st.session_state.show_lens_options = True
        lens_descriptions = []
        
        ordered_lenses = ['Monofocal'] + [lens for lens in prioritized_lenses if lens != 'Monofocal']
        
        for lens in ordered_lenses:
            description = get_lens_description(lens, user_lifestyle)
            if description:
                lens_descriptions.append(f"- {lens}: {description}")
        
        response = f"{st.session_state.doctor_name} has suggested the following lenses for you. I'd be happy to explain how each of these options might fit into your lifestyle. Please feel free to ask any questions you might have about these lenses - I'm here to help!\n\n"
        response += "\n\n".join(lens_descriptions)
        response += "\n\nIs there a particular lens you'd like to know more about?"
        return response

    if any(keyword in query.lower() for keyword in ["what lenses", "which lenses", "doctor suggest", "doctor recommend", "surgeon suggest", "surgeon recommend","clinic suggest", "clinic advise","clinic recomendation"]):
        lens_descriptions = []
        
        ordered_lenses = ['Monofocal'] + [lens for lens in prioritized_lenses if lens != 'Monofocal']
        
        for lens in ordered_lenses:
            description = get_lens_description(lens, user_lifestyle)
            if description:
                lens_descriptions.append(f"- {lens}: {description}")
        
        response = f"Dr. {st.session_state.doctor_name} has thoughtfully suggested the following lenses for you. I'd be happy to explain how each of these options might fit into your lifestyle. Please feel free to ask any questions you might have about these lenses - I'm here to help!\n\n"
        response += "\n\n".join(lens_descriptions)
        response += "\n\nPlease remember, these suggestions are tailored to your unique needs. If you have any specific questions about these lenses, don't hesitate to ask!"
        return response

    comparison_keywords = ["compare", "comparison", "difference", "versus", "vs"]
    is_comparison = any(keyword in query.lower() for keyword in comparison_keywords)
    
    lens_types = ["monofocal", "multifocal", "toric", "light adjustable"]
    specific_lens_query = next((lens for lens in lens_types if lens.lower() in query.lower()), None)
    
    if is_comparison:
        lenses_to_compare = [lens.capitalize() for lens in lens_types if lens.lower() in query.lower()]
        
        if len(lenses_to_compare) < 2:
            return "I'm sorry, but I couldn't identify which specific lens types you want to compare. Could you please clarify which lens types you'd like me to compare?"

    langchain_answer, _ = query_knowledge_base(query, vectorstore)
    
    if langchain_answer:
        refined_response = refine_langchain_response(langchain_answer, query, prioritized_lenses, specific_lens_query)
        return refined_response
    
    return chat_with_gpt(st.session_state.messages)

def refine_langchain_response(langchain_answer, user_query, prioritized_lenses, specific_lens_query=None):
    gpt_prompt = f"""
    Refine this answer about IOL lenses, making it conversational and easy to understand:
    
    Query: {user_query}
    Answer: {langchain_answer}
    {'Specific lens:' if specific_lens_query else 'Prioritized lenses:'} {specific_lens_query if specific_lens_query else ', '.join(prioritized_lenses)}

    Guidelines:
    - Use simple language and short sentences
    - Focus on the most relevant information
    {'- Focus EXCLUSIVELY on the specified lens type' if specific_lens_query else '- Prioritize information about the listed lenses'}
    - Don't make recommendations
    - Encourage consulting an eye doctor
    - Limit the response to about 100 words
    - Use bullet points for clarity

    {'Ensure that the response focuses SOLELY on the specified lens type and does not include information about other types of lenses unless it\'s crucial for understanding the specified lens.' if specific_lens_query else ''}

    Provide a concise, user-friendly response:
    """
    gpt_messages = [{"role": "user", "content": gpt_prompt}]
    return chat_with_gpt(gpt_messages)

def get_lens_description(lens_name, user_lifestyle):
    gpt_prompt = f"""
    Briefly describe the {lens_name} intraocular lens (IOL) for a patient with this lifestyle: {user_lifestyle}. 
    
    Guidelines:
    - Focus on how it fits daily activities or vision needs
    - Use 1-2 short sentences only
    - Be precise and concise
    - Don't recommend; only describe
    - Limit to 25 words maximum

    Provide a short description:
    """
    gpt_messages = [{"role": "user", "content": gpt_prompt}]
    return chat_with_gpt(gpt_messages)

def read_file(file):
    try:
        content = file.getvalue().decode("utf-8")
        data = content.splitlines()
        if len(data) < 2:
            st.error("The uploaded file does not contain enough information.")
            return None, None
        doctor_name = data[0].split(':')[1].strip()
        lenses = data[1].split(':')[1].strip().split(',')
        return doctor_name, [lens.strip() for lens in lenses]
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None

def fix_spelling(query):
    if len(query) > 100:
        return query

    prompt = f"""
    Please carefully review the following text for spelling errors only. Do not make any other changes.
    If there are no spelling errors, return the original text exactly as is.
    If there are spelling errors, correct them and return the corrected text.
    Do not add any explanations or comments.

    Text to review: {query}

    Corrected text:
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that corrects spelling errors without changing the meaning or intent of the text."},
        {"role": "user", "content": prompt}
    ]
    corrected_query = chat_with_gpt(messages)
    
    if len(corrected_query) != len(query) and abs(len(corrected_query) - len(query)) > 5:
        return query
    
    return corrected_query.strip()

def extract_name(input_text):
    prompt = f"""
    Please extract the name from the following input. If there's no clear name, respond with "None".
    Only provide the extracted name or "None", nothing else.

    Input: {input_text}

    Extracted name:
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that extracts names from text."},
        {"role": "user", "content": prompt}
    ]
    extracted_name = chat_with_gpt(messages)
    return None if extracted_name.strip().lower() == "none" else extracted_name.strip()

def main():
    st.set_page_config(page_title="AI-ASSISTANT FOR IOL EDUCATION", layout="wide")
    
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: #FFFFFF;
    }
    .chat-bubble {
        padding: 10px 15px;
        border-radius: 20px;
        margin-bottom: 10px;
        display: inline-block;
        max-width: 70%;
        word-wrap: break-word;
    }
    .bot-bubble {
        background-color: #2C3E50;
        float: left;
        clear: both;
    }
    .user-bubble {
        background-color: #4CAF50;
        float: right;
        clear: both;
    }
    .chat-container {
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("AI-Assistant for IOL Education")

    vectorstore = load_vectorstore()

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'user_lifestyle' not in st.session_state:
        st.session_state.user_lifestyle = ""
    if 'prioritized_lenses' not in st.session_state:
        st.session_state.prioritized_lenses = []
    if 'show_lens_options' not in st.session_state:
        st.session_state.show_lens_options = False
    if 'greeted' not in st.session_state:
        st.session_state.greeted = False
    if 'doctor_name' not in st.session_state:
        st.session_state.doctor_name = ""
    if 'asked_name' not in st.session_state:
        st.session_state.asked_name = False
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0

    uploaded_file = st.file_uploader("Please upload the file your surgeon has provided you", type=["txt"])

    if uploaded_file is not None and not st.session_state.greeted:
        doctor_name, prioritized_lenses = read_file(uploaded_file)
        if doctor_name and prioritized_lenses:
            st.session_state.doctor_name = doctor_name
            st.session_state.prioritized_lenses = prioritized_lenses
            initial_greeting = f"Hello! I'm {doctor_name}'s virtual assistant. I'm here to help you learn more about intraocular lenses (IOLs). I know this process can feel a bit overwhelming, but don't worry – we'll take it step by step together!"
            name_request = "Before we begin, I'd love to know your name. What should I call you?"
            
            st.session_state.messages = [
                {"role": "system", "content": "You are an AI assistant for IOL education."},
                {"role": "assistant", "content": initial_greeting},
                {"role": "assistant", "content": name_request}
            ]
            st.session_state.chat_history = [
                ("bot", initial_greeting),
                ("bot", name_request)
            ]
            st.session_state.greeted = True
            st.session_state.asked_name = True
        else:
            st.error("Unable to process the uploaded file. Please check the file format.")

    chat_container = st.container()
    with chat_container:
        for role, message in st.session_state.chat_history:
            if role == "bot":
                st.markdown(f"""
                <div class="chat-bubble bot-bubble">
                {message}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-bubble user-bubble">
                {message}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)

    if st.session_state.greeted:
        if 'input_key' not in st.session_state:
            st.session_state.input_key = 0

        with st.form(key='message_form'):
            user_input = st.text_input("You:", key=f"user_input_{st.session_state.input_key}")
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner("Processing your input..."):
                corrected_input = fix_spelling(user_input)
                
                st.session_state.messages.append({"role": "user", "content": corrected_input})
                st.session_state.chat_history.append(("user", corrected_input))
                st.session_state.question_count += 1
                
                print(f"Question count: {st.session_state.question_count}")

                if st.session_state.asked_name and not st.session_state.user_name:
                    extracted_name = extract_name(corrected_input)
                    if extracted_name:
                        st.session_state.user_name = extracted_name
                        bot_response = f"It's wonderful to meet you, {st.session_state.user_name}! Thank you for sharing your name with me. I'm here to help you learn more about IOLs and answer any questions you might have."
                        st.session_state.messages.append({"role": "assistant", "content": bot_response})
                        st.session_state.chat_history.append(("bot", bot_response))
                        
                        lifestyle_question = "Now, I'd love to get to know you better. Could you share a little bit about your lifestyle and your activities? This will help me understand your vision needs and how we can best support them. Feel free to tell me about your work, hobbies, or any visual tasks that are important to you!"
                        st.session_state.messages.append({"role": "assistant", "content": lifestyle_question})
                        st.session_state.chat_history.append(("bot", lifestyle_question))
                    else:
                        bot_response = "I'm sorry, I didn't catch your name. Could you please tell me your name again?"
                        st.session_state.messages.append({"role": "assistant", "content": bot_response})
                        st.session_state.chat_history.append(("bot", bot_response))
                else:
                    if not st.session_state.show_lens_options:
                        st.session_state.user_lifestyle = corrected_input
                    
                    bot_response = process_query(corrected_input, vectorstore, st.session_state.user_lifestyle, st.session_state.prioritized_lenses)

                    if bot_response:
                        st.session_state.messages.append({"role": "assistant", "content": bot_response})
                        st.session_state.chat_history.append(("bot", bot_response))

                        if st.session_state.question_count >= 5 and st.session_state.question_count % 2 == 1:
                            follow_up = "Is there anything else you'd like to know about IOLs or eye surgery?"
                            st.session_state.messages.append({"role": "assistant", "content": follow_up})
                            st.session_state.chat_history.append(("bot", follow_up))
                            
                            print(f"Follow-up prompt added to chat history (Question {st.session_state.question_count})")
                    else:
                        st.error("Sorry, I couldn't generate a response. Please try again.")

            st.session_state.input_key += 1
            st.experimental_rerun()

if __name__ == "__main__":
    main()