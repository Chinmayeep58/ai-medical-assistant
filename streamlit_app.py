import streamlit as st
import assemblyai as aai
import re
import json
import tempfile
import os
from typing import Dict, List, Tuple, Set, Optional
import logging
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from audio_recorder_streamlit import audio_recorder

# Configure logging
logging.basicConfig(level=logging.INFO)

class MedicalTranscriptAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the Medical Transcript Analyzer with AssemblyAI API key"""
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()
        
        # LeMUR prompts for different categories
        self.lemur_prompts = {
            'phi': """
            Analyze the following medical transcript and identify all Protected Health Information (PHI).
            PHI includes: names, ages, dates of birth, addresses, phone numbers, email addresses, 
            social security numbers, medical record numbers, account numbers, certificate/license numbers,
            vehicle identifiers, device identifiers, web URLs, IP addresses, biometric identifiers,
            full face photos, and any other unique identifying numbers or characteristics.
            
            Return a JSON array of objects with the following structure:
            [{"text": "identified_phi", "start_char": 0, "end_char": 10, "category": "name"}]
            
            Transcript: {transcript}
            """,
            
            'medical_entities': """
            Analyze the following medical transcript and identify medical entities in these categories:
            
            1. Medical Conditions/History: Any illnesses, diseases, symptoms, diagnoses, or medical conditions
            2. Anatomy: Body parts, organs, anatomical locations, or anatomical systems
            3. Medications: Prescribed drugs, over-the-counter medications, vitamins, supplements, drug classes
            4. Tests/Treatments/Procedures: Medical tests, treatments, procedures, therapies, surgeries, or interventions
            
            Return a JSON array of objects with the following structure:
            [{"text": "identified_term", "start_char": 0, "end_char": 10, "category": "medical_condition|anatomy|medication|procedure"}]
            
            Be thorough and include medical abbreviations, brand names, generic names, and medical terminology.
            
            Transcript: {transcript}
            """
        }

    def transcribe_audio(self, audio_file_path: str) -> Dict:
        """Transcribe audio file using AssemblyAI with enhanced features"""
        config = aai.TranscriptionConfig(
            entity_detection=True,
            punctuate=True,
            format_text=True,
            speaker_labels=True,
            auto_highlights=True,
            sentiment_analysis=True,
            custom_spelling=[
                {"from": ["hi tension"], "to": "hypertension"},
                {"from": ["die beetus"], "to": "diabetes"},
                {"from": ["ammonia"], "to": "pneumonia"},
                {"from": ["migrain"], "to": "migraine"},
            ]
        )
        
        transcript = self.transcriber.transcribe(audio_file_path, config)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Transcription failed: {transcript.error}")
        
        return {
            'transcript_id': transcript.id,
            'text': transcript.text,
            'entities': transcript.entities if transcript.entities else [],
            'utterances': transcript.utterances if hasattr(transcript, 'utterances') else [],
            'auto_highlights': transcript.auto_highlights_result if hasattr(transcript, 'auto_highlights_result') else None,
            'sentiment_analysis': transcript.sentiment_analysis_results if hasattr(transcript, 'sentiment_analysis_results') else None
        }

    def analyze_with_lemur(self, transcript_id: str, text: str) -> Dict:
        """Use AssemblyAI's LeMUR to analyze the transcript for medical entities"""
        results = {}
        
        try:
            # Analyze PHI using LeMUR
            phi_prompt = self.lemur_prompts['phi'].format(transcript=text)
            phi_response = aai.Lemur().task(
                phi_prompt,
                transcript_ids=[transcript_id],
                final_model=aai.LemurModel.claude3_5_sonnet
            )
            
            # Parse PHI response
            try:
                phi_data = json.loads(phi_response.response)
                results['phi'] = phi_data if isinstance(phi_data, list) else []
            except json.JSONDecodeError:
                results['phi'] = self._extract_json_from_response(phi_response.response)
            
            # Analyze medical entities using LeMUR
            medical_prompt = self.lemur_prompts['medical_entities'].format(transcript=text)
            medical_response = aai.Lemur().task(
                medical_prompt,
                transcript_ids=[transcript_id],
                final_model=aai.LemurModel.claude3_5_sonnet
            )
            
            # Parse medical entities response
            try:
                medical_data = json.loads(medical_response.response)
                results['medical_entities'] = medical_data if isinstance(medical_data, list) else []
            except json.JSONDecodeError:
                results['medical_entities'] = self._extract_json_from_response(medical_response.response)
            
        except Exception as e:
            logging.error(f"LeMUR analysis failed: {str(e)}")
            results = {'phi': [], 'medical_entities': []}
        
        return results

    def generate_medical_summary(self, transcript_id: str) -> str:
        """Generate a medical summary using LeMUR"""
        summary_prompt = """
        Create a concise medical summary of this patient consultation transcript. Include:
        1. Chief complaint or reason for visit
        2. Key symptoms mentioned by the patient
        3. Medical history discussed
        4. Current medications mentioned
        5. Tests or procedures discussed
        6. Treatment plan or recommendations given
        7. Follow-up instructions
        
        Format the summary in a professional medical style with clear sections.
        Keep it concise but comprehensive.
        """
        
        try:
            response = aai.Lemur().task(
                summary_prompt,
                transcript_ids=[transcript_id],
                final_model=aai.LemurModel.claude3_5_sonnet
            )
            return response.response
        except Exception as e:
            logging.error(f"Summary generation failed: {str(e)}")
            return "Summary generation failed."

    def ask_medical_question(self, transcript_id: str, question: str) -> str:
        """Ask a specific question about the medical transcript using LeMUR"""
        try:
            response = aai.Lemur().question(
                question,
                transcript_ids=[transcript_id],
                final_model=aai.LemurModel.claude3_5_sonnet
            )
            return response.response
        except Exception as e:
            logging.error(f"Question answering failed: {str(e)}")
            return "Unable to answer the question."

    def _extract_json_from_response(self, response_text: str) -> List:
        """Extract JSON array from LeMUR response text"""
        try:
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return []

    def format_transcript(self, text: str, entities: List, lemur_analysis: Dict) -> str:
        """Format the transcript with HTML highlighting"""
        # Extract PHI entities
        phi_entities = self.extract_phi_entities(entities, lemur_analysis.get('phi', []))
        
        # Extract medical entities from LeMUR
        medical_entities = self.extract_medical_entities(lemur_analysis.get('medical_entities', []))
        
        # Create formatting instructions
        format_instructions = []
        
        # Add PHI formatting
        for start, end, term in phi_entities:
            format_instructions.append((start, end, 'phi', term))
        
        # Add medical entity formatting
        for category, entity_set in medical_entities.items():
            for start, end, term in entity_set:
                format_instructions.append((start, end, category, term))
        
        # Sort by start position (reverse order for replacement)
        format_instructions.sort(key=lambda x: x[0], reverse=True)
        
        # Remove overlapping annotations
        filtered_instructions = []
        used_positions = set()
        
        for start, end, category, term in format_instructions:
            position_range = set(range(start, end))
            if not position_range.intersection(used_positions):
                filtered_instructions.append((start, end, category, term))
                used_positions.update(position_range)
        
        # Apply formatting
        formatted_text = text
        for start, end, category, term in filtered_instructions:
            if start >= len(formatted_text) or end > len(formatted_text):
                continue
                
            original_term = formatted_text[start:end]
            
            if category == 'phi':
                replacement = f'<span style="color: red; font-weight: bold;">{original_term}</span>'
            elif category == 'medical_condition':
                replacement = f'<span style="background-color: lightgreen; padding: 2px; border-radius: 3px;">{original_term}</span>'
            elif category == 'anatomy':
                replacement = f'<em style="background-color: lightblue; padding: 2px; border-radius: 3px;">{original_term}</em>'
            elif category == 'medication':
                replacement = f'<span style="background-color: yellow; padding: 2px; border-radius: 3px;">{original_term}</span>'
            elif category == 'procedure':
                replacement = f'<span style="color: darkblue; font-weight: bold; background-color: lightcyan; padding: 2px; border-radius: 3px;">{original_term}</span>'
            else:
                replacement = original_term
            
            formatted_text = formatted_text[:start] + replacement + formatted_text[end:]
        
        return formatted_text

    def extract_phi_entities(self, entities: List, lemur_phi: List) -> Set[Tuple[int, int, str]]:
        """Extract PHI entities from both AssemblyAI entities and LeMUR analysis"""
        phi_entities = set()
        
        # AssemblyAI detected entities
        phi_types = {
            'person_name', 'person_age', 'organization', 'location', 
            'phone_number', 'email_address', 'credit_card_number',
            'date_of_birth', 'drivers_license', 'passport_number'
        }
        
        for entity in entities:
            if entity.entity_type.lower() in phi_types:
                phi_entities.add((entity.start, entity.end, entity.text))
        
        # LeMUR detected PHI
        for phi_item in lemur_phi:
            if all(key in phi_item for key in ['text', 'start_char', 'end_char']):
                start = phi_item['start_char']
                end = phi_item['end_char']
                text = phi_item['text']
                phi_entities.add((start, end, text))
        
        return phi_entities

    def extract_medical_entities(self, lemur_medical: List) -> Dict[str, Set[Tuple[int, int, str]]]:
        """Extract medical entities from LeMUR analysis"""
        entities = {
            'medical_condition': set(),
            'anatomy': set(),
            'medication': set(),
            'procedure': set()
        }
        
        for item in lemur_medical:
            if all(key in item for key in ['text', 'start_char', 'end_char', 'category']):
                start = item['start_char']
                end = item['end_char']
                text = item['text']
                category = item['category']
                
                if category in entities:
                    entities[category].add((start, end, text))
        
        return entities

    def process_audio_file(self, audio_file_path: str) -> Dict:
        """Complete pipeline: transcribe audio, analyze with LeMUR, and format transcript"""
        # Transcribe audio
        transcription_result = self.transcribe_audio(audio_file_path)
        
        # Analyze with LeMUR
        lemur_analysis = self.analyze_with_lemur(
            transcription_result['transcript_id'], 
            transcription_result['text']
        )
        
        # Format transcript
        formatted_transcript = self.format_transcript(
            transcription_result['text'],
            transcription_result['entities'],
            lemur_analysis
        )
        
        # Generate summary
        medical_summary = self.generate_medical_summary(transcription_result['transcript_id'])
        
        return {
            'transcript_id': transcription_result['transcript_id'],
            'original_text': transcription_result['text'],
            'formatted_transcript': formatted_transcript,
            'entities': transcription_result['entities'],
            'lemur_analysis': lemur_analysis,
            'utterances': transcription_result['utterances'],
            'auto_highlights': transcription_result['auto_highlights'],
            'medical_summary': medical_summary,
            'sentiment_analysis': transcription_result['sentiment_analysis']
        }


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Medical Voice Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🏥 Medical Voice Assistant")
    st.markdown("""
    **AI-Powered Medical Transcript Analysis**
    
    This application helps healthcare providers analyze patient consultations by:
    - 🎤 Recording or uploading audio consultations
    - 📝 Converting speech to text with medical accuracy
    - 🔍 Identifying medical entities (conditions, medications, procedures)
    - 📊 Generating comprehensive medical summaries
    - ❓ Answering specific questions about consultations
    """)
    
    # Sidebar for API key and settings
    st.sidebar.header("⚙️ Configuration")
    api_key = st.sidebar.text_input(
        "AssemblyAI API Key", 
        type="password",
        help="Get your API key from https://www.assemblyai.com/"
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your AssemblyAI API key in the sidebar to continue.")
        st.info("💡 Get your free API key from [AssemblyAI](https://www.assemblyai.com/)")
        st.stop()
    
    # Initialize analyzer
    try:
        analyzer = MedicalTranscriptAnalyzer(api_key)
    except Exception as e:
        st.error(f"❌ Failed to initialize analyzer: {str(e)}")
        st.stop()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎤 Voice Recording", "📁 File Upload", "📊 Analysis Results", "❓ Q&A Assistant"])
    
    with tab1:
        st.header("🎤 Voice Recording")
        st.markdown("Click the microphone button below to start recording your consultation:")
        
        # Audio recorder
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#34495e",
            icon_name="microphone",
            icon_size="2x"
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            if st.button("🔄 Process Recording", type="primary"):
                with st.spinner("Processing audio... This may take a few minutes."):
                    try:
                        # Save audio bytes to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_bytes)
                            tmp_file_path = tmp_file.name
                        
                        # Process the audio
                        result = analyzer.process_audio_file(tmp_file_path)
                        
                        # Store result in session state
                        st.session_state['analysis_result'] = result
                        st.session_state['current_transcript_id'] = result['transcript_id']
                        
                        # Clean up temporary file
                        os.unlink(tmp_file_path)
                        
                        st.success("✅ Audio processed successfully! Check the 'Analysis Results' tab.")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing audio: {str(e)}")
    
    with tab2:
        st.header("📁 File Upload")
        st.markdown("Upload an audio file of your medical consultation:")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
            help="Supported formats: WAV, MP3, M4A, FLAC, OGG"
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file)
            
            if st.button("🔄 Process Uploaded File", type="primary"):
                with st.spinner("Processing audio file... This may take a few minutes."):
                    try:
                        # Save uploaded file to temporary location
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Process the audio
                        result = analyzer.process_audio_file(tmp_file_path)
                        
                        # Store result in session state
                        st.session_state['analysis_result'] = result
                        st.session_state['current_transcript_id'] = result['transcript_id']
                        
                        # Clean up temporary file
                        os.unlink(tmp_file_path)
                        
                        st.success("✅ File processed successfully! Check the 'Analysis Results' tab.")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing file: {str(e)}")
    
    with tab3:
        st.header("📊 Analysis Results")
        
        if 'analysis_result' not in st.session_state:
            st.info("ℹ️ No analysis results yet. Please record or upload audio first.")
        else:
            result = st.session_state['analysis_result']
            
            # Medical Summary
            st.subheader("📋 Medical Summary")
            st.markdown(result['medical_summary'])
            
            # Formatted Transcript
            st.subheader("📝 Annotated Transcript")
            st.markdown("**Legend:**")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown('<span style="color: red; font-weight: bold;">🔴 PHI (Personal Info)</span>', unsafe_allow_html=True)
            with col2:
                st.markdown('<span style="background-color: lightgreen; padding: 2px; border-radius: 3px;">🟢 Medical Conditions</span>', unsafe_allow_html=True)
            with col3:
                st.markdown('<em style="background-color: lightblue; padding: 2px; border-radius: 3px;">🔵 Anatomy</em>', unsafe_allow_html=True)
            with col4:
                st.markdown('<span style="background-color: yellow; padding: 2px; border-radius: 3px;">🟡 Medications</span>', unsafe_allow_html=True)
            with col5:
                st.markdown('<span style="color: darkblue; font-weight: bold; background-color: lightcyan; padding: 2px; border-radius: 3px;">🔷 Procedures</span>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown(result['formatted_transcript'], unsafe_allow_html=True)
            
            # Entity Statistics
            st.subheader("📈 Entity Statistics")
            lemur_analysis = result.get('lemur_analysis', {})
            
            # Create statistics
            entity_stats = {}
            for entity in lemur_analysis.get('medical_entities', []):
                category = entity.get('category', 'unknown')
                entity_stats[category] = entity_stats.get(category, 0) + 1
            
            phi_count = len(lemur_analysis.get('phi', []))
            if phi_count > 0:
                entity_stats['PHI'] = phi_count
            
            if entity_stats:
                # Create bar chart
                df_stats = pd.DataFrame(list(entity_stats.items()), columns=['Category', 'Count'])
                fig = px.bar(df_stats, x='Category', y='Count', 
                           title='Medical Entities Detected',
                           color='Category',
                           color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment Analysis
            if result.get('sentiment_analysis'):
                st.subheader("😊 Sentiment Analysis")
                sentiment_data = result['sentiment_analysis']
                
                if sentiment_data:
                    # Create sentiment timeline
                    sentiment_df = pd.DataFrame([
                        {
                            'Time': f"{s.start/1000:.1f}s",
                            'Sentiment': s.sentiment.title(),
                            'Confidence': s.confidence,
                            'Text': s.text[:50] + "..." if len(s.text) > 50 else s.text
                        }
                        for s in sentiment_data
                    ])
                    
                    st.dataframe(sentiment_df, use_container_width=True)
            
            # Auto Highlights
            if result.get('auto_highlights'):
                st.subheader("✨ Key Highlights")
                highlights = result['auto_highlights']
                if hasattr(highlights, 'results'):
                    for highlight in highlights.results[:5]:  # Show top 5
                        st.markdown(f"- **{highlight.text}** (Confidence: {highlight.rank:.2f})")
    
    with tab4:
        st.header("❓ Q&A Medical Assistant")
        
        if 'current_transcript_id' not in st.session_state:
            st.info("ℹ️ No transcript available. Please process audio first.")
        else:
            st.markdown("Ask specific questions about the medical consultation:")
            
            # Predefined questions
            st.subheader("🔍 Quick Questions")
            quick_questions = [
                "What medications were mentioned?",
                "What are the main symptoms discussed?",
                "What procedures or tests were recommended?",
                "What is the patient's chief complaint?",
                "What follow-up instructions were given?",
                "Were there any allergies mentioned?",
                "What is the patient's medical history?",
                "What was the diagnosis or assessment?"
            ]
            
            selected_question = st.selectbox("Choose a quick question:", [""] + quick_questions)
            
            if selected_question and st.button("Ask Quick Question"):
                with st.spinner("Getting answer..."):
                    answer = analyzer.ask_medical_question(
                        st.session_state['current_transcript_id'], 
                        selected_question
                    )
                    st.markdown(f"**Q:** {selected_question}")
                    st.markdown(f"**A:** {answer}")
            
            st.markdown("---")
            
            # Custom question
            st.subheader("💭 Custom Question")
            custom_question = st.text_area("Ask your own question about the consultation:")
            
            if custom_question and st.button("Ask Custom Question"):
                with st.spinner("Getting answer..."):
                    answer = analyzer.ask_medical_question(
                        st.session_state['current_transcript_id'], 
                        custom_question
                    )
                    st.markdown(f"**Q:** {custom_question}")
                    st.markdown(f"**A:** {answer}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🏥 Medical Voice Assistant - Powered by AssemblyAI LLM Models</p>
        <p><small>⚠️ This tool is for educational purposes. Always consult healthcare professionals for medical decisions.</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()