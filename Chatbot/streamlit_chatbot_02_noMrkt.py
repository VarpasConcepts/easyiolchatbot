import os
import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI as LangChainOpenAI
import time
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from io import BytesIO

# Add a debug flag
DEBUG = False

# Debugging function
def debug_print(message):
    if DEBUG:
        st.session_state.chat_history.append(("debug", f"DEBUG: {message}"))

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI client
client = OpenAI()

# Directory where the FAISS index is stored
storage_directory = "faiss_index"

@st.cache_resource
def load_vectorstore():
    debug_print("Entering load_vectorstore()")
    embeddings = OpenAIEmbeddings()
    try:
        index_path = os.path.join(os.path.dirname(__file__), storage_directory)
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        debug_print("Vectorstore loaded successfully")
        return vectorstore
    except Exception as e:
        st.error(f"Error loading index: {e}")
        st.error("The FAISS index file is missing or cannot be accessed. Please check the file path and permissions.")
        debug_print(f"Error in load_vectorstore(): {e}")
        return None

def query_knowledge_base(query, vectorstore):
    debug_print(f"Entering query_knowledge_base() with query: {query}")
    if vectorstore is None:
        debug_print("Vectorstore is None, returning None, None")
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
        debug_print("Query successful, returning result")
        return result["result"], result["source_documents"]
    except Exception as e:
        st.error(f"Error querying knowledge base: {e}")
        debug_print(f"Error in query_knowledge_base(): {e}")
        return None, None

def chat_with_gpt(messages):
    debug_print(f"Entering chat_with_gpt() with {len(messages)} messages")
    try:
        # Add a system message to encourage concise responses
        system_message = {
            "role": "system",
            "content": f'''
                "You are a friendly and empathetic assistant designed to help cataract patients understand intraocular lens (IOL) options. Your primary goals are to:
                    Provide clear, concise information about IOLs (aim for 50-100 words per response).
                    It is very important that you relate all information to the user's lifestyle as much as possible.
                    When information about a particular lens is asked, list out the definition, the pros and the cons of that lens.
                    When asked to compare between two or more lens types, give out the pros and cons of the lens relating them to the user's lifestyle information.
                    Use simple language, avoiding medical jargon when possible.
                    Encourage patients to ask questions for better understanding.
                    Never recommend specific IOLs or treatments.
                    Always advise consulting their ophthalmologist for personalized recommendations.

                Keep your tone warm and supportive. If a patient seems confused or hesitant, offer to explain things differently. Emphasize the importance of making informed decisions based on lifestyle needs and doctor's advice. If asked about specific IOL recommendations, politely redirect the patient to their doctor.
                Remember, your role is to educate and support, not to make medical decisions. Prioritize patient understanding and comfort in every interaction."
                '''
        }
        # Insert the system message at the beginning of the messages list
        messages.insert(0, system_message)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7  # Slightly increase randomness to encourage varied, concise responses
        )
        time.sleep(1)  # Add delay to prevent hitting rate limits
        debug_print("ChatGPT response received successfully")
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating ChatGPT response: {e}")
        debug_print(f"Error in chat_with_gpt(): {e}")
        return None

def is_marketing_appropriate(query):
    debug_print(f"Entering is_marketing_appropriate() with query: {query}")
    prompt = f"""
    Determine if the following query about IOL lenses is appropriate for a marketing-focused response:

    Query: {query}

    A marketing-focused response would be appropriate if the query:
    1. Asks about specific lens features or benefits
    2. Inquires about lens brands or products
    3. Seeks information on how lenses might improve quality of life
    4. Asks about the advantages of certain lens technologies

    It would NOT be appropriate if the query:
    1. Asks for medical advice
    2. Inquires about surgical procedures
    3. Asks about potential risks or complications
    4. Seeks personal recommendations
    5. Asks about pricing or insurance coverage

    ALWAYS REPLY FALSE TO THIS PROMPT
    """
    # Respond with only 'Yes' or 'No'.
    messages = [
        {"role": "system", "content": "You are an AI assistant that determines if queries are appropriate for marketing responses."},
        {"role": "user", "content": prompt}
    ]
    response = chat_with_gpt(messages)
    debug_print(f"is_marketing_appropriate() returning: {response.strip().lower() == 'yes'}")
    return response.strip().lower() == 'yes'

def process_query(query, vectorstore, user_lifestyle, prioritized_lenses):
    debug_print(f"Entering process_query() with query: {query}")
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
        debug_print("Returning initial lens options response")
        return response

    # Check if the query is about doctor's lens suggestions
    if any(keyword in query.lower() for keyword in ["what lenses", "which lenses", "doctor suggest", "doctor recommend", "surgeon suggest", "surgeon recommend","clinic suggest", "clinic advise","clinic recomendation", "gave me"]):
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
        debug_print("Returning doctor's lens suggestions response")
        return response

    # Check if this is a comparison query
    comparison_keywords = ["compare", "comparison", "difference", "versus", "vs"]
    is_comparison = any(keyword in query.lower() for keyword in comparison_keywords)
    
    # Check if this is a query about a specific lens type
    lens_types = ["monofocal", "multifocal", "toric", "light adjustable"]
    specific_lens_query = next((lens for lens in lens_types if lens.lower() in query.lower()), None)
    
    if is_comparison:
        # Identify which lenses are being compared
        lenses_to_compare = [lens.capitalize() for lens in lens_types if lens.lower() in query.lower()]
        
        if len(lenses_to_compare) < 2:
            debug_print("Insufficient lenses for comparison")
            return "I'm sorry, but I couldn't identify which specific lens types you want to compare. Could you please clarify which lens types you'd like me to compare?"

    # Process the query
    if is_marketing_appropriate(query):
        if specific_lens_query:
            langchain_answer, _ = query_knowledge_base(f"Provide information only about {specific_lens_query} IOL lenses", vectorstore)
        else:
            langchain_answer, _ = query_knowledge_base(query, vectorstore)
        
        if langchain_answer:
            refined_response = refine_langchain_response(langchain_answer, query, prioritized_lenses, specific_lens_query)
            debug_print("Returning merged response")
            return merge_responses(refined_response, query, user_lifestyle, prioritized_lenses, vectorstore, is_comparison, lenses_to_compare if is_comparison else None, specific_lens_query)
    
    # If not marketing appropriate or LangChain doesn't provide an answer, use ChatGPT
    debug_print("Using ChatGPT for response")
    return chat_with_gpt(st.session_state.messages)

def refine_langchain_response(langchain_answer, user_query, prioritized_lenses, specific_lens_query=None):
    debug_print(f"Entering refine_langchain_response() with query: {user_query}")
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
    refined_response = chat_with_gpt(gpt_messages)
    debug_print("Langchain response refined")
    return refined_response

def get_product_example(lens_type, vectorstore):
    debug_print(f"Entering get_product_example() for lens type: {lens_type}")
    query = f"Briefly name an example of a {lens_type} IOL product without recommending it. Use 10 words or less."
    result, _ = query_knowledge_base(query, vectorstore)
    debug_print(f"Product example retrieved: {result}")
    return result if result else ""

def merge_responses(langchain_refined, user_query, user_lifestyle, prioritized_lenses, vectorstore, is_comparison=False, lenses_to_compare=None, specific_lens_query=None):
    debug_print(f"Entering merge_responses() with query: {user_query}")
    if is_comparison and lenses_to_compare:
        lens_types = lenses_to_compare
    elif specific_lens_query:
        lens_types = [specific_lens_query.capitalize()]
    else:
        lens_types = ['Monofocal', 'Multifocal', 'Toric', 'Light Adjustable']
    
    mentioned_lenses = [lt for lt in lens_types if lt.lower() in langchain_refined.lower()]
    
    if specific_lens_query:
        product_examples = f"{specific_lens_query.capitalize()}: {get_product_example(specific_lens_query, vectorstore)}"
    else:
        product_examples = "; ".join([f"{lens}: {get_product_example(lens, vectorstore)}" for lens in mentioned_lenses])
    
    merge_prompt = f"""
    Create a concise response to this IOL lens query:
    
    Query: {user_query}
    Refined answer: {langchain_refined}
    User lifestyle: {user_lifestyle}
    {'Lenses to compare' if is_comparison else 'Specific lens' if specific_lens_query else 'Prioritized lenses'}: {', '.join(lenses_to_compare if is_comparison else [specific_lens_query.capitalize()] if specific_lens_query else prioritized_lenses)}
    Product examples: {product_examples}

    Guidelines:
    1. {'Focus on comparing ONLY the specified lens types' if is_comparison else 'Focus EXCLUSIVELY on the specified lens type' if specific_lens_query else 'Focus on relevant lens types and characteristics'}
    2. {'Briefly mention product examples without recommending' if not specific_lens_query else 'Mention ONLY the product example for the specified lens type without recommending'}
    3. Relate to user's lifestyle and activities
    4. Use simple language and short sentences
    5. Limit to 200 words maximum
    6. Use bullet points for clarity
    7. Don't make recommendations
    8. Encourage consulting an eye doctor

    {'Provide a clear comparison between ONLY the specified lens types, highlighting key differences and similarities.' if is_comparison else 'Provide detailed information EXCLUSIVELY about the specified lens type. DO NOT mention other lens types unless absolutely necessary for context.' if specific_lens_query else 'Provide a concise, informative response about the relevant lens types.'}
    
    If the query is about a specific lens type, ensure that the response focuses SOLELY on that lens type and does not include information about other types of lenses unless it's crucial for understanding the specified lens.
    """
    merge_messages = [{"role": "user", "content": merge_prompt}]
    merged_response = chat_with_gpt(merge_messages)
    debug_print("Responses merged successfully")
    return merged_response

def get_lens_description(lens_name, user_lifestyle):
    debug_print(f"Entering get_lens_description() for lens: {lens_name}")
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
    description = chat_with_gpt(gpt_messages)
    debug_print(f"Lens description generated for {lens_name}")
    return description

def read_file(file):
    debug_print("Entering read_file()")
    try:
        content = file.getvalue().decode("utf-8")
        data = content.splitlines()
        if len(data) < 2:
            st.error("The uploaded file does not contain enough information.")
            debug_print("File does not contain enough information")
            return None, None
        doctor_name = data[0].split(':')[1].strip()
        lenses = data[1].split(':')[1].strip().split(',')
        debug_print(f"File read successfully. Doctor: {doctor_name}, Lenses: {lenses}")
        return doctor_name, [lens.strip() for lens in lenses]
    except Exception as e:
        st.error(f"Error reading file: {e}")
        debug_print(f"Error in read_file(): {e}")
        return None, None

def fix_spelling(query):
    debug_print(f"Entering fix_spelling() with query: {query}")
    # If the query is longer than a certain threshold, skip spell-checking
    if len(query) > 100:
        debug_print("Query too long, skipping spell-check")
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
    
    # If the corrected query is significantly different, return the original
    if len(corrected_query) != len(query) and abs(len(corrected_query) - len(query)) > 5:
        debug_print("Significant difference in corrected query, returning original")
        return query
    
    debug_print(f"Spell-check complete. Corrected query: {corrected_query}")
    return corrected_query.strip()

def generate_summary(chat_history):
    debug_print("Entering generate_summary()")
    summary_prompt = f"""
    Please provide a concise summary of the following chat history between an AI assistant and a patient discussing intraocular lens (IOL) options. Focus on:

    1. The patient's main concerns and questions about IOLs
    2. Any specific lens types the patient showed interest in
    3. Key lifestyle factors that might influence lens choice
    4. Any misconceptions or areas where the patient needed clarification

    Chat History:
    {chat_history}

    Please provide a summary that would be helpful for the surgeon to quickly understand the patient's needs and concerns:
    """
    messages = [{"role": "user", "content": summary_prompt}]
    summary = chat_with_gpt(messages)
    debug_print("Summary generated")
    return summary

def create_pdf(chat_history, summary):
    debug_print("Entering create_pdf()")
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    Story = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

    # Add title
    Story.append(Paragraph("IOL Consultation Summary", styles['Heading1']))
    Story.append(Spacer(1, 12))

    # Add chat history
    Story.append(Paragraph("Chat History:", styles['Heading2']))
    for role, message in chat_history:
        p = Paragraph(f"<b>{role.capitalize()}:</b> {message}", styles['Justify'])
        Story.append(p)
        Story.append(Spacer(1, 6))

    Story.append(Spacer(1, 12))

    # Add summary
    Story.append(Paragraph("Patient Summary:", styles['Heading2']))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(summary, styles['Justify']))

    doc.build(Story)
    pdf_content = buffer.getvalue()
    buffer.close()
    
    debug_print("PDF created successfully")
    return pdf_content

def get_binary_file_downloader_html(bin_file, file_label='File'):
    debug_print("Entering get_binary_file_downloader_html()")
    b64 = base64.b64encode(bin_file).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_label}.pdf">Download {file_label}</a>'
    debug_print("Download link created")
    return href

def main():
    st.set_page_config(
        page_title="EasyIOLChat",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None
    )
    
    # Set the theme to light mode and add custom sidebar styling
    st.markdown("""
        <style>
        :root {
            --secondary-background-color: #f0f2f6;
        }
        [data-testid=stSidebar] {
            color:white;
            background-color: #092247;
        }
        .sidebar .sidebar-content {
            color: white;
        }
        .sidebar .sidebar-content {
            background-image: url('https://github.com/VarpasConcepts/easyiolchatbot/blob/main/easyiol.png');
            background-size: cover;
            background-repeat: no-repeat;
            background-position: top left;
            color: white;
        }
        .sidebar .sidebar-content .block-container {
            padding-top: 5rem;
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
            background-color: #D3D3D3;
            float: left;
            clear: both;
        }
        .user-bubble {
            background-color: #87CEFA;
            float: right;
            clear: both;
        }
        .debug-bubble {
            background-color: #FFB6C1;
            float: left;
            clear: both;
            font-style: italic;
        }
        .chat-container {
            margin-bottom: 20px;
        }
        }

        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create a sidebar
    with st.sidebar:
        # Add some space to push content below the logo
        st.empty()
        st.empty()
        st.empty()


    # Initialize session state variables
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
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0

    debug_print("Initializing vectorstore")
    vectorstore = load_vectorstore()

    uploaded_file = st.sidebar.file_uploader("Please upload the file your surgeon has provided you", type=["txt"])

    if uploaded_file is not None and not st.session_state.greeted:
        debug_print("File uploaded, processing")
        doctor_name, prioritized_lenses = read_file(uploaded_file)
        if doctor_name and prioritized_lenses:
            st.session_state.doctor_name = doctor_name
            st.session_state.prioritized_lenses = prioritized_lenses
            initial_greeting = f"Hello! I'm {doctor_name}'s virtual assistant. I'm here to help you navigate the world of intraocular lenses (IOLs) and find the perfect fit for your lifestyle. I know this process can feel a bit overwhelming, but don't worry – we'll take it step by step together!"
            name_request = "Before we begin, I'd love to know your name. What should I call you?"
            
            st.session_state.messages = [
                {"role": "system", "content": "You are an AI assistant for IOL selection."},
                {"role": "assistant", "content": initial_greeting},
                {"role": "assistant", "content": name_request}
            ]
            st.session_state.chat_history = [
                ("bot", initial_greeting),
                ("bot", name_request)
            ]
            st.session_state.greeted = True
            st.session_state.asked_name = True
            debug_print("Greeting and name request set")
        else:
            st.error("Unable to process the uploaded file. Please check the file format.")
            debug_print("Error processing uploaded file")

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
            elif role == "user":
                st.markdown(f"""
                <div class="chat-bubble user-bubble">
                {message}
                </div>
                """, unsafe_allow_html=True)
            elif role == "debug":
                st.markdown(f"""
                <div class="chat-bubble debug-bubble">
                {message}
                </div>
                """, unsafe_allow_html=True)
        
        # Add some space after the chat bubbles
        st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)

    if st.session_state.greeted:
        if st.session_state.asked_name and not st.session_state.user_name:
            # Display two input fields for first name and last name
            with st.form(key='name_form'):
                col1, col2 = st.columns(2)
                with col1:
                    first_name = st.text_input("First Name:", key="first_name")
                with col2:
                    last_name = st.text_input("Last Name:", key="last_name")
                
                submit_button = st.form_submit_button("Submit")

            if submit_button or (first_name and last_name):  # This allows both button click and Enter key to submit
                if first_name and last_name:
                    st.session_state.user_name = f"{first_name} {last_name}"
                    
                    # Add the user's name to the chat history
                    st.session_state.messages.append({"role": "user", "content": st.session_state.user_name})
                    st.session_state.chat_history.append(("user", st.session_state.user_name))
                    
                    bot_response = f"It's wonderful to meet you, {st.session_state.user_name}! Thank you so much for sharing your name with me. I'm excited to help you learn more about IOLs and find the best option for your unique needs."
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                    st.session_state.chat_history.append(("bot", bot_response))
                    
                    lifestyle_question = "Now, I'd love to get to know you better. Could you share a little bit about your lifestyle and your activities? This will help me understand your vision needs and how we can best support them. Feel free to tell me about your work, hobbies, or any visual tasks that are important to you!"
                    st.session_state.messages.append({"role": "assistant", "content": lifestyle_question})
                    st.session_state.chat_history.append(("bot", lifestyle_question))
                    debug_print(f"User name set and added to chat history: {st.session_state.user_name}")
                    st.session_state.asked_name = False
                    st.experimental_rerun()  # Force a rerun to update the UI
                else:
                    st.warning("Please enter both your first and last name.")
        else:
            with st.form(key='message_form'):
                # Use a unique key for the text input field
                user_input = st.text_input("You:", key=f"user_input_{st.session_state.input_key}")
                col1, col2 = st.columns([3, 1])
                with col1:
                    submit_button = st.form_submit_button(label='Send')
                with col2:
                    end_conversation_button = st.form_submit_button(label='End Conversation')

            if submit_button and user_input:
                debug_print(f"Processing user input: {user_input}")
                with st.spinner("Processing your input..."):
                    # Apply spell-checking in the background
                    corrected_input = fix_spelling(user_input)
                    debug_print(f"Corrected input: {corrected_input}")
                    
                    st.session_state.messages.append({"role": "user", "content": corrected_input})
                    st.session_state.chat_history.append(("user", corrected_input))
                    st.session_state.question_count += 1
                    
                    debug_print(f"Question count: {st.session_state.question_count}")

                    if not st.session_state.show_lens_options:
                        st.session_state.user_lifestyle = corrected_input
                        bot_response = process_query(corrected_input, vectorstore, st.session_state.user_lifestyle, st.session_state.prioritized_lenses)
                        st.session_state.messages.append({"role": "assistant", "content": bot_response})
                        st.session_state.chat_history.append(("bot", bot_response))
                        debug_print("Processed initial lifestyle query")
                    else:
                        bot_response = process_query(corrected_input, vectorstore, st.session_state.user_lifestyle, st.session_state.prioritized_lenses)

                        if bot_response:
                            st.session_state.messages.append({"role": "assistant", "content": bot_response})
                            st.session_state.chat_history.append(("bot", bot_response))

                            # Check if it's time to display the follow-up prompt (5th question and every odd question after)
                            if st.session_state.question_count >= 5 and st.session_state.question_count % 2 == 1:
                                follow_up = "I want to make sure you have all the information you need about IOLs. Is there anything else you're curious about or would like me to explain further?"
                                st.session_state.messages.append({"role": "assistant", "content": follow_up})
                                st.session_state.chat_history.append(("bot", follow_up))
                                debug_print(f"Follow-up prompt added to chat history (Question {st.session_state.question_count})")
                        else:
                            st.error("Sorry, I couldn't generate a response. Please try again.")
                            debug_print("Failed to generate bot response")

                # Increment the input key to force a reset of the input field
                st.session_state.input_key += 1
                st.experimental_rerun()

            if end_conversation_button:
                debug_print("End conversation button clicked")
                with st.spinner("Generating conversation summary..."):
                    # Generate summary
                    chat_history_text = "\n".join([f"{role}: {message}" for role, message in st.session_state.chat_history])
                    summary = generate_summary(chat_history_text)
                    
                    # Create PDF
                    pdf_content = create_pdf(st.session_state.chat_history, summary)
                    
                    # Provide download link
                    st.markdown(get_binary_file_downloader_html(pdf_content, 'IOL_Consultation_Summary'), unsafe_allow_html=True)
                    
                    # Clear conversation state
                    st.session_state.messages = []
                    st.session_state.chat_history = []
                    st.session_state.user_lifestyle = ""
                    st.session_state.show_lens_options = False
                    st.session_state.greeted = False
                    st.session_state.asked_name = False
                    st.session_state.user_name = ""
                    st.session_state.question_count = 0
                    st.session_state.input_key = 0
                    
                    st.success("Thank you for your time. Your consultation summary is ready for download.")
                    debug_print("Conversation ended, summary generated, and state reset")

if __name__ == "__main__":
    main()