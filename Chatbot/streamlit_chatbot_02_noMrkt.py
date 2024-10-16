import os
import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI as LangChainOpenAI
import time
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from io import BytesIO
import re
from pathlib import Path
from pydub import AudioSegment
import uuid
import tempfile

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
    
    
def format_and_replace(text, doctor_name):
    # Replace doctor-related words with the actual doctor's name + office
    doctor_words = ['doctor', 'surgeon', 'ophthalmologist']
    for word in doctor_words:
        # Remove "your" (case-insensitive) before the doctor's name
        text = re.sub(r'\b(?i:your\s+)?' + word + r'\b', f"{doctor_name}'s office", text, flags=re.IGNORECASE)
    # Remove any remaining "your" before the doctor's name
    text = re.sub(r'\b(?i:your)\s+' + re.escape(f"{doctor_name}'s office"), f"{doctor_name}'s office", text)
    # Add line breaks for better readability
    text = text.replace(". ", ".\n")
    # Improve bullet point handling
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        if '•' in line:
            # Split the line at each bullet point
            bullet_points = line.split('•')
            # Add each bullet point as a separate line
            for point in bullet_points:
                if point.strip():  # Ignore empty strings
                    formatted_lines.append(f"  • {point.strip()}")
        else:
            formatted_lines.append(line)
    # Join the lines back together
    text = '\n'.join(formatted_lines)
    return text

def process_response(response, doctor_name):
    formatted_response = format_and_replace(response, doctor_name)
    return formatted_response

# Add this function for text-to-speech conversion
def text_to_speech(text, voice_type="alloy"):
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice_type,
            input=text
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(response.content)
            audio_file_path = tmp_file.name
        
        return audio_file_path
    except Exception as e:
        st.error("An error occurred during text-to-speech conversion.")
        return None

def display_chat_bubble(role, message):
    if role == "bot":
        message_hash = hash(message)
        st.markdown(f"""
        <div class="chat-bubble bot-bubble">
        {message}
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([0.02, 0.2, 0.2])
        with col1:
            if st.button("🔊", key=f"tts_button_{message_hash}"):
                speech_file_path = text_to_speech(message)
                if speech_file_path and os.path.exists(speech_file_path):
                    with open(speech_file_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                    st.session_state['current_audio'] = {
                        'hash': message_hash,
                        'audio': audio_bytes
                    }

        with col2:
            if 'current_audio' in st.session_state and isinstance(st.session_state['current_audio'], dict):
                current_audio = st.session_state['current_audio']
                if current_audio.get('hash') == message_hash and 'audio' in current_audio:
                    st.audio(current_audio['audio'], format="audio/mp3")
        
        with col3:
            if 'current_audio' in st.session_state and isinstance(st.session_state['current_audio'], dict):
                current_audio = st.session_state['current_audio']
                if current_audio.get('hash') == message_hash and 'audio' in current_audio:
                    st.download_button(
                        label="Download",
                        data=current_audio['audio'],
                        file_name="speech.mp3",
                        mime="audio/mp3",
                        key=f"download_button_{message_hash}"
                    )
    elif role == "user":
        st.markdown(f"""
        <div class="chat-bubble user-bubble">
        {message}
        </div>
        """, unsafe_allow_html=True)

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
        # Add a system message to encourage nurse-like responses
        system_message = {
            "role": "system",
            "content": f'''
                    "Hi there! You're a friendly nurse at our cataract surgery clinic. Your job is to chat with patients about intraocular lenses (IOLs) in a warm, caring way. Your responses will be delivered both as text and through text-to-speech, so keep this in mind as you communicate. Here's what we need you to do:

                    Keep things simple and easy to understand. Aim for about 50-100 words per reply.
                    Use natural, conversational language that sounds good when spoken aloud.
                    Include brief pauses in your speech by using commas and periods appropriately.
                    Avoid complex sentence structures or long lists that might be hard to follow when spoken.
                    Use contractions (e.g., "you're" instead of "you are") to sound more natural in speech.
                    Try to relate the IOL info to the patient's day-to-day life when you can.
                    If someone asks about a specific lens, give them a quick rundown of what it is, plus the good and not-so-good points.
                    When comparing lenses, talk about the pros and cons in terms of how they might fit into the patient's lifestyle but don't use the words pros or cons, use something like this can be advantageous to you or this can be disadvantageous to you.
                    Avoid medical jargon - explain things like you're chatting with a friend.
                    Use analogies or simple examples to help patients visualize complex concepts.
                    Encourage patients to speak up if they're unsure about anything. We want them to feel comfortable asking questions!
                    Remember, we can't recommend specific IOLs or treatments. That's the ophthalmologist's job.
                    Always remind patients to talk to their ophthalmologist for personalized advice.
                    Always refer to the eye doctor or surgeon or doctor as 'ophthalmologist' only and nothing else.

                    Super important: We never say one lens is better than another. Your role is to give the facts about each lens type without pushing the patient towards any particular choice. Steer clear of anything that sounds like you're recommending a specific lens.
                    Keep your tone warm and friendly, as if you're having a face-to-face conversation. If a patient seems confused, offer to explain things in a different way. Emphasize how important it is for them to make informed decisions based on their lifestyle and their ophthalmologist's advice. If they ask for specific IOL recommendations, gently remind them that their ophthalmologist is the best person to help with that decision.
                    Remember, you're here to educate and support, not to make medical decisions. Your main goal is to help patients understand their options and feel comfortable, while staying neutral about the different lens choices.
                    For speech output:

                    Speak at a moderate pace, neither too fast nor too slow.
                    Use a warm, caring tone of voice.
                    Vary your intonation slightly to maintain interest and emphasize key points.
                    Include brief, natural pauses between sentences and thoughts.
                    Pronounce medical terms clearly, but follow them with simple explanations.

                    You've got this! Let's help our patients feel informed and cared for, both through text and speech!"
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

def generate_potential_questions(current_context, chat_history):
    prompt = f"""
    Based on the following conversation context and chat history, generate 3-5 potential follow-up questions that the user might want to ask about intraocular lenses (IOLs). 
    These questions should be relevant, concise, and diverse.

    Current context: {current_context}
    Chat history: {chat_history}

    Generate the questions in the following format:
    1. Question 1
    2. Question 2
    3. Question 3
    ...

    Questions:
    """
    messages = [{"role": "user", "content": prompt}]
    response = chat_with_gpt(messages)
    
    # Parse the response into a list of questions
    questions = [q.split('. ', 1)[1] for q in response.split('\n') if q.strip() and q[0].isdigit()]
    
    return questions

def process_query_existing(query, vectorstore, user_lifestyle, prioritized_lenses):
    debug_print(f"Entering process_query_existing() with query: {query}")
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
            return process_response("I'm sorry, but I couldn't identify which specific lens types you want to compare. Could you please clarify which lens types you'd like me to compare?", st.session_state.doctor_name)

    # Process the query
    if is_marketing_appropriate(query):
        if specific_lens_query:
            langchain_answer, _ = query_knowledge_base(f"Provide information only about {specific_lens_query} IOL lenses", vectorstore)
        else:
            langchain_answer, _ = query_knowledge_base(query, vectorstore)
        
        if langchain_answer:
            refined_response = refine_langchain_response(langchain_answer, query, prioritized_lenses, specific_lens_query)
            debug_print("Returning merged response")
            merged_response = merge_responses(refined_response, query, user_lifestyle, prioritized_lenses, vectorstore, is_comparison, lenses_to_compare if is_comparison else None, specific_lens_query)
            return process_response(merged_response, st.session_state.doctor_name)
    
    # If not marketing appropriate or LangChain doesn't provide an answer, use ChatGPT
    debug_print("Using ChatGPT for response")
    chatgpt_response = chat_with_gpt(st.session_state.messages)
    return process_response(chatgpt_response, st.session_state.doctor_name)

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

def get_product_example(lens_type, vectorstore):
    debug_print(f"Entering get_product_example() for lens type: {lens_type}")
    query = f"Briefly name an example of a {lens_type} IOL product without recommending it. Use 10 words or less."
    result, _ = query_knowledge_base(query, vectorstore)
    debug_print(f"Product example retrieved: {result}")
    return result if result else ""

def process_query(query, vectorstore, user_lifestyle, prioritized_lenses):
    debug_print(f"Entering process_query() with query: {query}")
    # Check if this is the first response after user shares their lifestyle
    if not st.session_state.show_lens_options:
        st.session_state.show_lens_options = True
        
        # Thank the user for sharing lifestyle details
        thank_you_response = f"Thank you for sharing those details about your lifestyle, {st.session_state.user_name}. It's really helpful to understand your daily activities and visual needs."
        
        # Add message about cataract surgery and ask if they want to know more about IOLs
        cataract_message = "I understand that cataract surgery can feel overwhelming, but don't worry - we'll take it step by step together. Would you like to know more about intraocular lenses (IOLs) before we look at the options your surgeon has suggested for you?"
        
        return [thank_you_response, cataract_message]

    # Check if the user wants to know more about IOLs
    if query.lower() == "yes":
        iols_explanation = chat_with_gpt([
            {"role": "system", "content": "You are an AI assistant explaining IOLs to a patient."},
            {"role": "user", "content": f"Explain what intraocular lenses (IOLs) are, relating the explanation to this lifestyle: {user_lifestyle}"}
        ])
        
        # Separate message for lens suggestions
        lens_suggestions = "Based on your lifestyle and visual needs, " + st.session_state.doctor_name + " has suggested the following lenses for you:\n\n"
        for lens in prioritized_lenses:
            description = get_lens_description(lens, user_lifestyle)
            if description:
                lens_suggestions += f"• {lens}: {description}\n\n"
        lens_suggestions += "Which lens would you like to know more about?"
        
        return [iols_explanation, lens_suggestions]
    
    elif query.lower() == "no":
        lens_suggestions = "Certainly!" + st.session_state.doctor_name + " has suggested the following lenses based on your lifestyle and visual needs:\n\n"
        for lens in prioritized_lenses:
            description = get_lens_description(lens, user_lifestyle)
            if description:
                lens_suggestions += f"• {lens}: {description}\n\n"
        lens_suggestions += "Which lens would you like to know more about?"
        
        return [lens_suggestions]
    
    # For all other queries, use the existing logic
    bot_response = process_query_existing(query, vectorstore, user_lifestyle, prioritized_lenses)

    if bot_response and bot_response not in ["SHOW_BUTTONS", "WAIT_FOR_LENS_CHOICE"]:
        # Generate potential questions after every bot response
        current_context = f"User lifestyle: {user_lifestyle}\nPrioritized lenses: {', '.join(prioritized_lenses)}\nLast bot response: {bot_response}"
        chat_history = "\n".join([f"{role}: {content}" for role, content in st.session_state.chat_history[-5:]])  # Use last 5 messages for context
        potential_questions = generate_potential_questions(current_context, chat_history)
        st.session_state.potential_questions = potential_questions

        if st.session_state.question_count >= 5 and st.session_state.question_count % 2 == 1:
            follow_up = "I want to make sure you have all the information you need about IOLs. Is there anything else you're curious about or would like me to explain further?"
            return [bot_response, follow_up]
        else:
            return [bot_response]
    else:
        return bot_response

    return bot_response

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
    5. The patient's overall understanding and comfort level with the IOL options discussed

    Chat History:
    {chat_history}

    Please provide a summary that would be helpful for the surgeon to quickly understand the patient's needs, concerns, and the outcome of the consultation:
    """
    messages = [{"role": "user", "content": summary_prompt}]
    summary = chat_with_gpt(messages)
    debug_print("Summary generated")
    return summary

def format_text_for_pdf(text):
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        if line.strip().startswith('•'):
            # Remove the bullet point and add some indentation
            formatted_lines.append(f"    {line.strip()[1:].strip()}")
        elif ':' in line and not line.strip().endswith(':'):
            parts = line.split(':', 1)
            formatted_lines.append(f"<b>{parts[0].strip()}:</b>{parts[1]}")
        else:
            formatted_lines.append(line)
    return '<br/>'.join(formatted_lines)

def create_pdf(chat_history, summary, user_name, doctor_name, user_lifestyle, prioritized_lenses):
    debug_print("Entering create_pdf()")
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    Story = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='ChatBubble', 
                              fontSize=10,
                              leading=14,
                              spaceBefore=6,
                              spaceAfter=30,
                              leftIndent=20,
                              rightIndent=20,
                              allowWidows=0,
                              allowOrphans=0))
    styles.add(ParagraphStyle(name='UserBubble', 
                              parent=styles['ChatBubble'],
                              alignment=TA_RIGHT, 
                              textColor=colors.black,
                              backColor=colors.lightblue,
                              borderColor=colors.lightblue,
                              borderWidth=1,
                              borderPadding=(10,10,10,10),
                              borderRadius=10))
    styles.add(ParagraphStyle(name='BotBubble', 
                              parent=styles['ChatBubble'],
                              alignment=TA_LEFT, 
                              textColor=colors.black,
                              backColor=colors.lightgrey,
                              borderColor=colors.lightgrey,
                              borderWidth=1,
                              borderPadding=(10,10,10,10),
                              borderRadius=10))

    # Add title
    Story.append(Paragraph("IOL Consultation Summary", styles['Heading1']))
    Story.append(Spacer(1, 12))

    # Add patient summary
    Story.append(Paragraph("Patient Summary", styles['Heading2']))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph(f"Patient Name: {user_name}", styles['Normal']))
    Story.append(Paragraph(f"Referring Doctor: Dr. {doctor_name}", styles['Normal']))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph("Lifestyle and Visual Needs:", styles['Normal']))
    Story.append(Paragraph(format_text_for_pdf(user_lifestyle), styles['Justify']))
    Story.append(Spacer(1, 6))
    Story.append(Paragraph("Prioritized Lens Options:", styles['Normal']))
    Story.append(Paragraph(", ".join(prioritized_lenses), styles['Justify']))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph("Consultation Summary:", styles['Normal']))
    Story.append(Paragraph(format_text_for_pdf(summary), styles['Justify']))
    Story.append(Spacer(1, 12))

    # Add page break before chat history
    Story.append(PageBreak())

    # Add chat history
    Story.append(Paragraph("Detailed Conversation", styles['Heading2']))
    Story.append(Spacer(1, 12))
    for role, message in chat_history:
        formatted_message = format_text_for_pdf(message)
        if role == "user":
            p = Paragraph(formatted_message, styles['UserBubble'])
        elif role == "bot":
            p = Paragraph(formatted_message, styles['BotBubble'])
        Story.append(p)

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

def process_user_input(user_input, vectorstore):
    debug_print(f"Processing user input: {user_input}")
    with st.spinner("Processing your input..."):
        corrected_input = fix_spelling(user_input)
        debug_print(f"Corrected input: {corrected_input}")
        
        st.session_state.messages.append({"role": "user", "content": corrected_input})
        st.session_state.chat_history.append(("user", corrected_input))
        st.session_state.question_count += 1
        
        # Capture lifestyle information if it's the first response after asking for it
        if st.session_state.question_count == 1:
            st.session_state.user_lifestyle = corrected_input
        
        debug_print(f"Question count: {st.session_state.question_count}")

        bot_responses = process_query(corrected_input, vectorstore, st.session_state.user_lifestyle, st.session_state.prioritized_lenses)

        if bot_responses == "SHOW_BUTTONS" or bot_responses == "WAIT_FOR_LENS_CHOICE":
            st.experimental_rerun()
        elif bot_responses:
            for response in bot_responses:
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.chat_history.append(("bot", response))
        else:
            st.error("Sorry, I couldn't generate a response. Please try again.")
            debug_print("Failed to generate bot response")

    st.session_state.input_key += 1
    st.experimental_rerun()

def end_conversation():
    debug_print("End conversation button clicked")
    with st.spinner("Generating conversation summary..."):
        chat_history_text = "\n".join([f"{role}: {message}" for role, message in st.session_state.chat_history])
        summary = generate_summary(chat_history_text)
        
        pdf_content = create_pdf(
            st.session_state.chat_history, 
            summary, 
            st.session_state.user_name, 
            st.session_state.doctor_name, 
            st.session_state.user_lifestyle, 
            st.session_state.prioritized_lenses
        )
        
        st.markdown(get_binary_file_downloader_html(pdf_content, 'IOL_Consultation_Summary'), unsafe_allow_html=True)
        
        # Reset session state variables
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
            /* Root variables */
            :root {
                --secondary-background-color: #f0f2f6;
                font-size: 28px;
            }

            [data-testid="stSidebar"] [data-testid="stSidebarNav"] button[kind="header"] {
                color: white !important;
            }

            /* Sidebar styling */
            [data-testid=stSidebar] {
                background-color: #092247;
            }

            .sidebar .sidebar-content {
                color: white;
            }

            .sidebar .sidebar-content .block-container {
                padding-top: 1rem;
            }

            [data-testid=stSidebar] [data-testid=stText],
            [data-testid=stSidebar] [data-testid=stMarkdown] p {
                color: white !important;
            }

            /* File uploader styling */
            [data-testid="stFileUploader"] {
                margin-top: -2rem;
            }

            [data-testid="stFileUploader"] > label {
                color: white !important;
            }

            [data-testid="stFileUploader"] > div > div {
                background-color: white !important;
                border-radius: 5px !important;
            }

            [data-testid="stFileUploadDropzone"] {
                min-height: 100px !important;
            }

            [data-testid="stFileUploadDropzone"] button {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #cccccc !important;
            }

            .st-emotion-cache-16idsys p {
                word-break: break-word;
                margin-bottom: 0px;
                font-size: 20px;
            }

            /* Chat bubble styling */
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

            /* Button styling */
            .stButton > button {
                color: white !important;
                border: none !important;
                padding: 10px 24px !important;
                text-align: center !important;
                text-decoration: none !important;
                display: inline-block !important;
                font-size: 16px !important;
                margin: 4px 2px !important;
                cursor: pointer !important;
                border-radius: 5px !important;
            }

            /* "Yes, tell me more about IOLs" button (Green) */
            div.row-widget.stButton:first-of-type button {
                background-color: #4CAF50 !important; /* Green */
            }

            /* "No, show me the lens options" button (Red) */
            div.row-widget.stButton:nth-of-type(2) button {
                background-color: #FF4B4B !important; /* Red */
            }

            /* Specifically targeting the "End Conversation" button */
            div.row-widget.stButton.st-emotion-cache-k008qs button {
                background-color: #FF4B4B !important; /* Red */
                color: white !important;
            }

            /* Ensure the Send button retains its original style */
            div.row-widget.stButton button:not(.st-emotion-cache-k008qs) {
                background-color: #007bff !important; /* Default Send button color */
                color: white !important; /* White text for Send button */
            }

            /* Hover effects */
            .stButton > button:hover {
                opacity: 0.8 !important;
            }

            /* Fix for overriding container styles that may affect button layout */
            div.row-widget.stButton {
                display: flex;
                justify-content: flex-start !important;
            }

            /* Additional styling for sidebar elements if needed */
            .sidebar-text {
                color: white;
                font-size: 1rem;
                margin-bottom: 0.5rem;
            }
                
            /* Cross button styling */
            .st-emotion-cache-ztfqz8 {
                display: inline-flex;
                -webkit-box-align: center;
                align-items: center;
                -webkit-box-pack: center;
                justify-content: center;
                font-weight: 400;
                border-radius: 0.5rem;
                margin: 0px 0.125rem;
                color: white;
                width: auto;
                user-select: none;
                background-color: transparent;
                border: none;
                padding: 0.5rem;
                font-size: 14px;
                line-height: 1;
                min-width: 2rem;
                min-height: 2rem;
            }

            /* Updated styles for TTS buttons and audio player */
            .stAudio {
                margin-top: 0;
                margin-bottom: 0;
            }
            .stAudio > div {
                height: 20px;
                max-width: 150px;
            }
            .stAudio > div > div {
                height: 20px;
            }
            .stAudio > div > div > div > div > div {
                max-width: 150px;  /* Adjust this value as needed */
            }
            .stButton > button {
                margin-top: 0;
                height: 40px;
                padding: 0 10px !important;
            }
            .stDownloadButton > button {
                margin-top: 0;
                height: 40px;
                white-space: nowrap;
            }
            /* Align columns */
            [data-testid="column"] {
                display: flex;
                align-items: center;
            }
            /* Style for sound icon button */
            [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child .stButton > button {
                background-color: #0066FF !important;
                color: white !important;
            }
            /* Style for Download button */
            [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(4) .stDownloadButton > button {
                background-color: white !important;
                color: black !important;
                border: 2px solid black !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Create a sidebar
    with st.sidebar:
        # Add some space to push content below the logo
        st.empty()
        st.empty()
        st.sidebar.image("./easyiol.png", use_column_width=True)
        st.markdown("<p class='sidebar-text'>Please upload the file that was provided by your doctor</p>", unsafe_allow_html=True)

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
    if 'current_audio' not in st.session_state:
        st.session_state.current_audio = None

    debug_print("Initializing vectorstore")
    vectorstore = load_vectorstore()

    uploaded_file = st.sidebar.file_uploader("", type=["txt"])

    if uploaded_file is not None and not st.session_state.greeted:
        debug_print("File uploaded, processing")
        doctor_name, prioritized_lenses = read_file(uploaded_file)
        if doctor_name and prioritized_lenses:
            st.session_state.doctor_name = doctor_name
            st.session_state.prioritized_lenses = prioritized_lenses
            initial_greeting = f"Hello! I'm {doctor_name}'s virtual assistant. I'm here to help you navigate the world of intraocular lenses (IOLs) and find the perfect fit for your lifestyle. I know this process can feel a bit overwhelming, but don't worry – we'll take it step by step together!"
            name_request = "Before we begin, I'd love to know your name. What should I call you?"
            
            initial_greeting = process_response(initial_greeting, doctor_name)
            name_request = process_response(name_request, doctor_name)
            
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
            display_chat_bubble(role, message)
        
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

            if submit_button and first_name and last_name:
                st.session_state.user_name = f"{first_name} {last_name}"
                # Add the user's name to the chat history
                st.session_state.messages.append({"role": "user", "content": st.session_state.user_name})
                st.session_state.chat_history.append(("user", st.session_state.user_name))
                
                bot_response = f"It's so wonderful to meet you, {st.session_state.user_name}! Thank you for sharing your name with me. I'm really excited to help you explore IOLs and find the best option for your needs. Feel free to ask me anything about cataract surgery or intraocular lenses—just type your question in the chat! As we chat, I'll generate a few questions based on our conversation. If anything piques your curiosity, just click to learn more. I'm here to help every step of the way!"
                bot_response = process_response(bot_response, st.session_state.doctor_name)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                st.session_state.chat_history.append(("bot", bot_response))
                
                lifestyle_question = "I'd love to get to know you better. Could you share a little bit about your lifestyle and your activities? This will help me understand your vision needs and how we can best support them. Feel free to tell me about your work, hobbies, or any visual tasks that are important to you!"
                lifestyle_question = process_response(lifestyle_question, st.session_state.doctor_name)
                st.session_state.messages.append({"role": "assistant", "content": lifestyle_question})
                st.session_state.chat_history.append(("bot", lifestyle_question))
                debug_print(f"User name set and added to chat history: {st.session_state.user_name}")
                st.session_state.asked_name = False
                st.experimental_rerun()  # Force a rerun to update the UI
            elif submit_button:
                st.warning("Please enter both your first and last name.")
        else:
            # Check if we need to show the Yes/No buttons
            if st.session_state.show_lens_options and st.session_state.chat_history[-1][1].endswith("suggested for you?"):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, tell me more about IOLs"):
                        process_user_input("yes", vectorstore)
                with col2:
                    if st.button("No, show me the lens options"):
                        process_user_input("no", vectorstore)
            else:
                # Display potential questions as buttons
                if 'potential_questions' in st.session_state and st.session_state.potential_questions:
                    st.write("You might want to ask:")
                    for question in st.session_state.potential_questions:
                        if st.button(question):
                            process_user_input(question, vectorstore)

                # Display text input for custom questions
                with st.form(key='message_form'):
                    user_input = st.text_input("Or ask your own question:", key=f"user_input_{st.session_state.input_key}")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        submit_button = st.form_submit_button(label='Send')
                    with col2:
                        end_conversation_button = st.form_submit_button(label='End Conversation')

                if submit_button and user_input:
                    process_user_input(user_input, vectorstore)
                elif end_conversation_button:
                    end_conversation()

if __name__ == "__main__":
    main()