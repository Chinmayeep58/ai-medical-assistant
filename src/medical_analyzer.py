import assemblyai as aai
import re
import json
from typing import Dict, List, Tuple, Set, Optional
import logging

class MedicalTranscriptAnalyzer:
    def __init__(self, api_key: str):
        """
        Initialize the Medical Transcript Analyzer with AssemblyAI API key
        
        Args:
            api_key (str): Your AssemblyAI API key
        """
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

    def transcribe_audio(self, audio_file_path: str, enable_speaker_diarization: bool = True) -> Dict:
        """
        Transcribe audio file using AssemblyAI with enhanced features
        
        Args:
            audio_file_path (str): Path to the audio file or URL
            enable_speaker_diarization (bool): Enable speaker identification
            
        Returns:
            Dict: Transcription result with entities and metadata
        """
        config = aai.TranscriptionConfig(
            entity_detection=True,
            punctuate=True,
            format_text=True,
            speaker_labels=enable_speaker_diarization,
            auto_highlights=True,
            sentiment_analysis=True,
            iab_categories=True,
            content_safety=True,
            custom_spelling=[
                # Add common medical terms that might be misheard
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
            'sentiment_analysis': transcript.sentiment_analysis_results if hasattr(transcript, 'sentiment_analysis_results') else None,
            'iab_categories': transcript.iab_categories_result if hasattr(transcript, 'iab_categories_result') else None
        }

    def analyze_with_lemur(self, transcript_id: str, text: str) -> Dict:
        """
        Use AssemblyAI's LeMUR to analyze the transcript for medical entities
        
        Args:
            transcript_id (str): AssemblyAI transcript ID
            text (str): Transcript text
            
        Returns:
            Dict: Analysis results from LeMUR
        """
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
                # Fallback: extract JSON from response text
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
        """
        Generate a medical summary using LeMUR
        
        Args:
            transcript_id (str): AssemblyAI transcript ID
            
        Returns:
            str: Medical summary
        """
        summary_prompt = """
        Create a concise medical summary of this transcript. Include:
        1. Chief complaint or reason for visit
        2. Key symptoms mentioned
        3. Medical history discussed
        4. Medications mentioned
        5. Tests or procedures discussed
        6. Treatment plan or recommendations
        
        Format the summary in a professional medical style with clear sections.
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
        """
        Ask a specific question about the medical transcript using LeMUR
        
        Args:
            transcript_id (str): AssemblyAI transcript ID
            question (str): Question to ask about the transcript
            
        Returns:
            str: Answer to the question
        """
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
        """
        Extract JSON array from LeMUR response text
        
        Args:
            response_text (str): Raw response from LeMUR
            
        Returns:
            List: Extracted entities or empty list
        """
        try:
            # Find JSON array in the response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return []

    def extract_phi_entities(self, entities: List, lemur_phi: List) -> Set[Tuple[int, int, str]]:
        """
        Extract PHI entities from both AssemblyAI entities and LeMUR analysis
        
        Args:
            entities (List): Entities from AssemblyAI transcription
            lemur_phi (List): PHI entities from LeMUR analysis
            
        Returns:
            Set[Tuple[int, int, str]]: Combined PHI entities
        """
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
        """
        Extract medical entities from LeMUR analysis
        
        Args:
            lemur_medical (List): Medical entities from LeMUR
            
        Returns:
            Dict[str, Set[Tuple[int, int, str]]]: Categorized medical entities
        """
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

    def format_transcript(self, text: str, entities: List, lemur_analysis: Dict) -> str:
        """
        Format the transcript with HTML highlighting using both AssemblyAI and LeMUR results
        
        Args:
            text (str): Original transcript text
            entities (List): Entities detected by AssemblyAI
            lemur_analysis (Dict): Analysis results from LeMUR
            
        Returns:
            str: Formatted HTML text
        """
        # Extract PHI entities
        phi_entities = self.extract_phi_entities(
            entities, 
            lemur_analysis.get('phi', [])
        )
        
        # Extract medical entities from LeMUR
        medical_entities = self.extract_medical_entities(
            lemur_analysis.get('medical_entities', [])
        )
        
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
                replacement = f'<span style="color: red;">{original_term}</span>'
            elif category == 'medical_condition':
                replacement = f'<span style="background-color: lightgreen;">{original_term}</span>'
            elif category == 'anatomy':
                replacement = f'<em>{original_term}</em>'
            elif category == 'medication':
                replacement = f'<span style="background-color: yellow;">{original_term}</span>'
            elif category == 'procedure':
                replacement = f'<span style="color: darkblue;">{original_term}</span>'
            else:
                replacement = original_term
            
            formatted_text = formatted_text[:start] + replacement + formatted_text[end:]
        
        return formatted_text

    def process_audio_file(self, audio_file_path: str, generate_summary: bool = True) -> Dict:
        """
        Complete pipeline: transcribe audio, analyze with LeMUR, and format transcript
        
        Args:
            audio_file_path (str): Path to audio file or URL
            generate_summary (bool): Whether to generate a medical summary
            
        Returns:
            Dict: Complete analysis results
        """
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
        
        result = {
            'transcript_id': transcription_result['transcript_id'],
            'original_text': transcription_result['text'],
            'formatted_transcript': formatted_transcript,
            'entities': transcription_result['entities'],
            'lemur_analysis': lemur_analysis,
            'utterances': transcription_result['utterances'],
            'auto_highlights': transcription_result['auto_highlights']
        }
        
        # Generate summary if requested
        if generate_summary:
            result['medical_summary'] = self.generate_medical_summary(
                transcription_result['transcript_id']
            )
        
        return result

    def process_text_with_mock_transcript(self, text: str) -> Dict:
        """
        Process existing text by creating a mock transcript and using LeMUR
        
        Args:
            text (str): Text to process
            
        Returns:
            Dict: Analysis results
        """
        # For text-only processing, we create a temporary transcript
        # Note: This requires uploading the text as an audio file or using the submit method
        # For demonstration, we'll use a simplified approach
        
        try:
            # Submit text for analysis (you would need to convert text to audio or use API differently)
            # This is a placeholder - in practice you'd need to handle text-to-speech conversion
            # or use AssemblyAI's text analysis features if available
            
            # For now, we'll use basic pattern matching as fallback
            lemur_analysis = {'phi': [], 'medical_entities': []}
            
            formatted_text = self.format_transcript(text, [], lemur_analysis)
            
            return {
                'original_text': text,
                'formatted_transcript': formatted_text,
                'lemur_analysis': lemur_analysis
            }
            
        except Exception as e:
            logging.error(f"Text processing failed: {str(e)}")
            raise


# Example usage and testing
def main():
    """
    Example usage of the MedicalTranscriptAnalyzer
    """
    # Initialize with your AssemblyAI API key
    api_key = "your_assemblyai_api_key_here"
    analyzer = MedicalTranscriptAnalyzer(api_key)
    
    try:
        # Process audio file
        audio_file = "path_to_your_audio_file.wav"  # or URL
        result = analyzer.process_audio_file(audio_file)
        
        print("=== FORMATTED TRANSCRIPT ===")
        print(result['formatted_transcript'])
        
        print("\n=== MEDICAL SUMMARY ===")
        if 'medical_summary' in result:
            print(result['medical_summary'])
        
        # Ask specific questions about the transcript
        transcript_id = result['transcript_id']
        
        questions = [
            "What medications were mentioned in this conversation?",
            "What are the main symptoms discussed?",
            "Were any procedures or tests recommended?",
            "What is the patient's chief complaint?"
        ]
        
        print("\n=== Q&A ANALYSIS ===")
        for question in questions:
            answer = analyzer.ask_medical_question(transcript_id, question)
            print(f"Q: {question}")
            print(f"A: {answer}\n")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()


# Additional utility functions for web interface or API integration
class MedicalTranscriptWebInterface:
    """
    Web interface wrapper for the medical transcript analyzer
    """
    
    def __init__(self, api_key: str):
        self.analyzer = MedicalTranscriptAnalyzer(api_key)
    
    def upload_and_process(self, file_data, file_type: str = "audio") -> Dict:
        """
        Process uploaded file (audio or text)
        
        Args:
            file_data: File data or path
            file_type (str): "audio" or "text"
            
        Returns:
            Dict: Processing results
        """
        if file_type == "audio":
            return self.analyzer.process_audio_file(file_data)
        else:
            return self.analyzer.process_text_with_mock_transcript(file_data)
    
    def get_entity_statistics(self, analysis_result: Dict) -> Dict:
        """
        Get statistics about detected entities
        
        Args:
            analysis_result (Dict): Result from process_audio_file
            
        Returns:
            Dict: Entity statistics
        """
        lemur_analysis = analysis_result.get('lemur_analysis', {})
        
        stats = {
            'phi_count': len(lemur_analysis.get('phi', [])),
            'medical_entities': {}
        }
        
        for entity in lemur_analysis.get('medical_entities', []):
            category = entity.get('category', 'unknown')
            stats['medical_entities'][category] = stats['medical_entities'].get(category, 0) + 1
        
        return stats