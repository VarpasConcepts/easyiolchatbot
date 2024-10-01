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
        # Add a system message to encourage concise responses
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant providing information about IOLs. Please aim to keep your responses concise, ideally between 150-200 words. Cover all relevant information but prioritize brevity and clarity. If a longer response is absolutely necessary to adequately address the query, you may exceed this limit, but strive to be as concise as possible."
        }
        
        # Insert the system message at the beginning of the messages list
        messages.insert(0, system_message)
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=300,  # This is a soft limit, approximately 225 words
            temperature=0.7  # Slightly increase randomness to encourage varied, concise responses
        )
        time.sleep(1)  # Add delay to prevent hitting rate limits
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating ChatGPT response: {e}")
        return None

def is_marketing_appropriate(query):
    prompt = f"""
    Determine if the following query about IOL lenses is appropriate for a marketing-focused response:

    Query: {query}

    A marketing-focused response would be appropriate if the query:
    1. Asks about specific lens features or benefits
    2. Compares different types of lenses
    3. Inquires about lens brands or products
    4. Seeks information on how lenses might improve quality of life
    5. Asks about the advantages of certain lens technologies

    It would NOT be appropriate if the query:
    1. Asks for medical advice
    2. Inquires about surgical procedures
    3. Asks about potential risks or complications
    4. Seeks personal recommendations
    5. Asks about pricing or insurance coverage

    Respond with only 'Yes' or 'No'.
    """
    messages = [
        {"role": "system", "content": "You are an AI assistant that determines if queries are appropriate for marketing responses."},
        {"role": "user", "content": prompt}
    ]
    response = chat_with_gpt(messages)
    return response.strip().lower() == 'yes'

def process_query(query, vectorstore, user_lifestyle, prioritized_lenses):
    # Check if this is the first response after user shares their lifestyle
    if not st.session_state.show_lens_options:
        st.session_state.show_lens_options = True
        lens_descriptions = []
        
        # Ensure monofocal lens is always first
        ordered_lenses = ['Monofocal'] + [lens for lens in prioritized_lenses if lens != 'Monofocal']
        
        for lens in ordered_lenses:
            description = get_lens_description(lens, user_lifestyle)
            if description:
                lens_descriptions.append(f"- {lens}: {description}")
        
        response = f"{st.session_state.doctor_name} has suggested the following lenses for you. I'd be happy to explain how each of these options might fit into your lifestyle. Please feel free to ask any questions you might have about these lenses - I'm here to help!\n\n"
        response += "\n\n".join(lens_descriptions)
        response += "\n\nIs there a particular lens you'd like to know more about?"
        return response

    # Check if the query is about doctor's lens suggestions
    if any(keyword in query.lower() for keyword in ["what lenses", "which lenses", "doctor suggest", "doctor recommend"]):
        lens_descriptions = []
        
        # Ensure monofocal lens is always first
        ordered_lenses = ['Monofocal'] + [lens for lens in prioritized_lenses if lens != 'Monofocal']
        
        for lens in ordered_lenses:
            description = get_lens_description(lens, user_lifestyle)
            if description:
                lens_descriptions.append(f"- {lens}: {description}")
        
        response = f"Dr. {st.session_state.doctor_name} has thoughtfully suggested the following lenses for you. I'd be happy to explain how each of these options might fit into your lifestyle. Please feel free to ask any questions you might have about these lenses - I'm here to help!\n\n"
        response += "\n\n".join(lens_descriptions)
        response += "\n\nPlease remember, these suggestions are tailored to your unique needs. If you have any specific questions about these lenses, don't hesitate to ask!"
        return response

    # Check if this is a comparison query
    comparison_keywords = ["compare", "comparison", "difference", "versus", "vs"]
    is_comparison = any(keyword in query.lower() for keyword in comparison_keywords)
    
    if is_comparison:
        # Identify which lenses are being compared
        lenses_to_compare = []
        for lens in ["monofocal", "multifocal", "toric", "light adjustable"]:
            if lens in query.lower():
                lenses_to_compare.append(lens.capitalize())
        
        if len(lenses_to_compare) < 2:
            return "I'm sorry, but I couldn't identify which specific lens types you want to compare. Could you please clarify which lens types you'd like me to compare?"

    # Existing logic for other queries
    if is_marketing_appropriate(query):
        langchain_answer, _ = query_knowledge_base(query, vectorstore)
        if langchain_answer:
            refined_response = refine_langchain_response(langchain_answer, query, prioritized_lenses)
            return merge_responses(refined_response, query, user_lifestyle, prioritized_lenses, vectorstore, is_comparison, lenses_to_compare if is_comparison else None)
    
    # If not marketing appropriate or LangChain doesn't provide an answer, use ChatGPT
    return chat_with_gpt(st.session_state.messages)

def refine_langchain_response(langchain_answer, user_query, prioritized_lenses):
    gpt_prompt = f"""
    Refine this answer about IOL lenses, making it conversational and easy to understand:
    
    Query: {user_query}
    Answer: {langchain_answer}
    Prioritized lenses: {', '.join(prioritized_lenses)}

    Guidelines:
    - Use simple language and short sentences
    - Focus on the most relevant information
    - Prioritize information about the listed lenses
    - Don't make recommendations
    - Encourage consulting an eye doctor
    - Limit the response to about 100 words
    - Use bullet points for clarity

    Provide a concise, user-friendly response:
    """
    gpt_messages = [{"role": "user", "content": gpt_prompt}]
    return chat_with_gpt(gpt_messages)

def get_product_example(lens_type, vectorstore):
    query = f"Briefly name an example of a {lens_type} IOL product without recommending it. Use 10 words or less."
    result, _ = query_knowledge_base(query, vectorstore)
    return result if result else ""

def merge_responses(langchain_refined, user_query, user_lifestyle, prioritized_lenses, vectorstore, is_comparison=False, lenses_to_compare=None):
    if is_comparison and lenses_to_compare:
        lens_types = lenses_to_compare
    else:
        lens_types = ['Monofocal', 'Multifocal', 'Toric', 'Light Adjustable']
    
    mentioned_lenses = [lt for lt in lens_types if lt.lower() in langchain_refined.lower()]
    
    product_examples = "; ".join([f"{lens}: {get_product_example(lens, vectorstore)}" for lens in mentioned_lenses])
    
    merge_prompt = f"""
    Create a concise response to this IOL lens query:
    
    Query: {user_query}
    Refined answer: {langchain_refined}
    User lifestyle: {user_lifestyle}
    {'Lenses to compare' if is_comparison else 'Prioritized lenses'}: {', '.join(lenses_to_compare if is_comparison else prioritized_lenses)}
    Product examples: {product_examples}

    Guidelines:
    1. {'Focus on comparing the specified lens types' if is_comparison else 'Focus on relevant lens types and characteristics'}
    2. Briefly mention product examples without recommending
    3. Relate to user's lifestyle and activities
    4. Use simple language and short sentences
    5. Limit to 200 words maximum
    6. Use bullet points for clarity
    7. Don't make recommendations
    8. Encourage consulting an eye doctor

    {'Provide a clear comparison between the specified lens types, highlighting key differences and similarities.' if is_comparison else 'Provide a concise, informative response about the relevant lens types.'}
    """
    merge_messages = [{"role": "user", "content": merge_prompt}]
    return chat_with_gpt(merge_messages)

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
        if len(data) < 4:
            st.error("The uploaded file does not contain enough information.")
            return None, None, None, None
        doctor_name = data[0].split(':')[1].strip()
        name = data[1].split(':')[1].strip()
        age = data[2].split(':')[1].strip()
        lenses = data[3].split(':')[1].strip().split(',')
        return doctor_name, name, age, [lens.strip() for lens in lenses]
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None, None, None

def main():
    st.set_page_config(page_title="AI-ASSISTANT FOR IOL EDUCATION", layout="wide")
    
    # Set the theme to dark mode and add chat bubble styles
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

    uploaded_file = st.file_uploader("Upload the .txt file with patient details", type=["txt"])

    if uploaded_file is not None and not st.session_state.greeted:
        doctor_name, name, age, prioritized_lenses = read_file(uploaded_file)
        if doctor_name and name and age and prioritized_lenses:
            st.session_state.doctor_name = doctor_name
            st.session_state.prioritized_lenses = prioritized_lenses
            st.session_state.messages = [
                {"role": "system", "content": "You are an AI assistant for IOL selection."},
                {"role": "assistant", "content": f"Hi {name}! I'm {doctor_name}'s virtual assistant. I'm here to help you navigate the world of intraocular lenses (IOLs) and find the perfect fit for your lifestyle. I know this process can feel a bit overwhelming, but don't worry – we'll take it step by step together!"},
                {"role": "assistant", "content": f"Before we dive into the details about IOLs, I'd love to get to know you better. Could you share a little bit about your lifestyle and your activities? This will help me understand your vision needs and how we can best support them. Feel free to tell me about your work, hobbies, or any visual tasks that are important to you!"}
            ]
            st.session_state.chat_history = [
                ("bot", f"Hi {name}! I'm {doctor_name}'s virtual assistant. I'm here to help you navigate the world of intraocular lenses (IOLs) and find the perfect fit for your lifestyle. I know this process can feel a bit overwhelming, but don't worry – we'll take it step by step together!"),
                ("bot", f"Before we dive into the details about IOLs, I'd love to get to know you better. Could you share a little bit about your lifestyle and your activities? This will help me understand your vision needs and how we can best support them. Feel free to tell me about your work, hobbies, or any visual tasks that are important to you!")
            ]
            st.session_state.greeted = True
        else:
            st.error("Unable to process the uploaded file. Please check the file format.")

    # Chat bubble display
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
        
        # Add some space after the chat bubbles
        st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)

    if st.session_state.greeted:
        # Initialize the input key in session state if it doesn't exist
        if 'input_key' not in st.session_state:
            st.session_state.input_key = 0

        with st.form(key='message_form'):
            # Use a unique key for the text input field
            user_input = st.text_input("You:", key=f"user_input_{st.session_state.input_key}")
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append(("user", user_input))

            if not st.session_state.show_lens_options:
                st.session_state.user_lifestyle = user_input
                with st.spinner("Processing your information..."):
                    bot_response = process_query(user_input, vectorstore, st.session_state.user_lifestyle, st.session_state.prioritized_lenses)
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                    st.session_state.chat_history.append(("bot", bot_response))
            else:
                with st.spinner("Processing your question..."):
                    bot_response = process_query(user_input, vectorstore, st.session_state.user_lifestyle, st.session_state.prioritized_lenses)

                    if bot_response:
                        st.session_state.messages.append({"role": "assistant", "content": bot_response})
                        st.session_state.chat_history.append(("bot", bot_response))
                    else:
                        st.error("Sorry, I couldn't generate a response. Please try again.")

            # Increment the input key to force a reset of the input field
            st.session_state.input_key += 1
            st.experimental_rerun()

if __name__ == "__main__":
    main()