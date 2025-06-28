import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import re
from pathlib import Path
import warnings
import tempfile
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Automated Metadata Generation System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Import all the libraries from your notebook
@st.cache_resource
def load_models():
    """Load all NLP models - cached to avoid reloading"""
    try:
        # Document processing libraries
        import fitz  # PyMuPDF
        import pdfplumber
        import docx2txt
        import pytesseract
        from PIL import Image
        import cv2
        
        # NLP libraries
        import spacy
        from transformers import pipeline
        from keybert import KeyBERT
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize, sent_tokenize
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # Load spaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.error("spaCy English model not found. Please install it with: python -m spacy download en_core_web_sm")
            return None
        
        # Initialize models
        kw_model = KeyBERT()
        
        try:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            st.warning(f"Could not load summarization model: {e}")
            summarizer = None
        
        try:
            stop_words = set(stopwords.words('english'))
        except Exception as e:
            st.warning(f"Could not load stopwords: {e}")
            stop_words = set()
        
        return {
            'nlp': nlp,
            'kw_model': kw_model,
            'summarizer': summarizer,
            'stop_words': stop_words,
            'libraries': {
                'fitz': fitz,
                'pdfplumber': pdfplumber,
                'docx2txt': docx2txt,
                'pytesseract': pytesseract,
                'Image': Image,
                'cv2': cv2,
                'word_tokenize': word_tokenize,
                'sent_tokenize': sent_tokenize,
                'TfidfVectorizer': TfidfVectorizer
            }
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Load models
models = load_models()
if models is None:
    st.stop()

# Extract models and libraries for global use
nlp = models['nlp']
kw_model = models['kw_model']
summarizer = models['summarizer']
stop_words = models['stop_words']
libs = models['libraries']

# Include all your classes from the notebook here
class PDFTextExtractor:
    def __init__(self):
        self.extracted_metadata = {}
    
    def extract_with_pymupdf(self, pdf_path):
        """Extract text using PyMuPDF (fitz) library"""
        try:
            doc = libs['fitz'].open(pdf_path)
            text_content = []
            
            # Extract basic document metadata
            metadata = doc.metadata
            self.extracted_metadata = {
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', ''),
                'page_count': doc.page_count
            }
            
            # Extract text from each page
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    text_content.append(text)
            
            doc.close()
            return '\n'.join(text_content)
            
        except Exception as e:
            st.error(f"PyMuPDF extraction failed: {e}")
            return ""
    
    def extract_with_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber library"""
        try:
            text_content = []
            
            with libs['pdfplumber'].open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
            
            return '\n'.join(text_content)
            
        except Exception as e:
            st.error(f"pdfplumber extraction failed: {e}")
            return ""
    
    def extract_pdf_text(self, pdf_path):
        """Extract text using both methods and return the best result"""
        # Try PyMuPDF first
        pymupdf_text = self.extract_with_pymupdf(pdf_path)
        
        # Try pdfplumber as backup
        pdfplumber_text = self.extract_with_pdfplumber(pdf_path)
        
        # Use the method that extracted more text
        if len(pymupdf_text) > len(pdfplumber_text):
            selected_text = pymupdf_text
            method_used = "PyMuPDF"
        else:
            selected_text = pdfplumber_text
            method_used = "pdfplumber"
        
        return selected_text, self.extracted_metadata

class DOCXTextExtractor:
    def __init__(self):
        self.extracted_metadata = {}
    
    def extract_docx_text(self, docx_path):
        """Extract text from DOCX files using docx2txt"""
        try:
            # Extract text content
            text = libs['docx2txt'].process(docx_path)
            
            # Get basic file information
            file_stats = os.stat(docx_path)
            self.extracted_metadata = {
                'file_size': file_stats.st_size,
                'creation_time': file_stats.st_ctime,
                'modification_time': file_stats.st_mtime,
                'word_count': len(text.split()) if text else 0,
                'character_count': len(text) if text else 0
            }
            
            return text, self.extracted_metadata
            
        except Exception as e:
            st.error(f"DOCX extraction failed: {e}")
            return "", {}

class OCRTextExtractor:
    def __init__(self):
        self.extracted_metadata = {}
    
    def preprocess_image(self, image_path):
        """Preprocess image for better OCR results"""
        try:
            # Read image
            image = libs['cv2'].imread(image_path)
            
            # Convert to grayscale
            gray = libs['cv2'].cvtColor(image, libs['cv2'].COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blur = libs['cv2'].GaussianBlur(gray, (3, 3), 0)
            
            # Apply threshold
            thresh = libs['cv2'].threshold(blur, 0, 255, libs['cv2'].THRESH_BINARY_INV + libs['cv2'].THRESH_OTSU)[1]
            
            # Morphological operations to remove noise
            kernel = libs['cv2'].getStructuringElement(libs['cv2'].MORPH_RECT, (3, 3))
            opening = libs['cv2'].morphologyEx(thresh, libs['cv2'].MORPH_OPEN, kernel, iterations=1)
            
            # Invert image
            processed_image = 255 - opening
            
            return processed_image
            
        except Exception as e:
            st.error(f"Image preprocessing failed: {e}")
            return None
    
    def extract_ocr_text(self, image_path):
        """Extract text from images using OCR"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            if processed_image is None:
                return "", {}
            
            # Configure OCR settings
            ocr_config = '--psm 6'  # Assume uniform block of text
            
            # Extract text
            text = libs['pytesseract'].image_to_string(processed_image, config=ocr_config, lang='eng')
            
            # Get image metadata
            with libs['Image'].open(image_path) as img:
                self.extracted_metadata = {
                    'image_size': img.size,
                    'image_mode': img.mode,
                    'image_format': img.format,
                    'character_count': len(text) if text else 0
                }
            
            return text, self.extracted_metadata
            
        except Exception as e:
            st.error(f"OCR extraction failed: {e}")
            return "", {}

class TextPreprocessor:
    def __init__(self):
        self.stop_words = stop_words.copy()
        # Add custom stopwords for document processing
        self.custom_stopwords = {
            'page', 'pages', 'document', 'file', 'pdf', 'docx', 'txt',
            'figure', 'table', 'section', 'chapter', 'appendix'
        }
        self.stop_words.update(self.custom_stopwords)
    
    def basic_cleaning(self, text):
        """Perform basic text cleaning operations"""
        if not text:
            return ""
        
        # Remove extra whitespaces and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Remove or replace special characters
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        # Fix common OCR errors
        text = re.sub(r'\s+([\.|!?])', r'\1', text)  # Fix spacing before punctuation
        text = re.sub(r'([\.|!?])\s*([A-Z])', r'\1 \2', text)  # Fix spacing after punctuation
        
        # Remove excessive punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def remove_noise(self, text):
        """Remove document-specific noise and artifacts"""
        if not text:
            return ""
        
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
        
        # Remove URL patterns
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email patterns
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove excessive numbering patterns
        text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
        
        # Remove table-like structures
        text = re.sub(r'\t+', ' ', text)
        text = re.sub(r' {3,}', ' ', text)
        
        return text
    
    def normalize_text(self, text):
        """Normalize text for consistent processing"""
        if not text:
            return ""
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Expand contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Normalize currency and number formats
        text = re.sub(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', 'CURRENCY', text)
        text = re.sub(r'\b\d{4}\b', 'YEAR', text)  # Years
        text = re.sub(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', 'NUMBER', text)  # Other numbers
        
        return text
    
    def preprocess_text(self, text, remove_stopwords=False, normalize=True):
        """Complete preprocessing pipeline"""
        if not text:
            return {"processed_text": "", "sentences": [], "word_count": 0}
        
        # Apply cleaning pipeline
        processed_text = self.basic_cleaning(text)
        processed_text = self.remove_noise(processed_text)
        
        if normalize:
            processed_text = self.normalize_text(processed_text)
        
        # Extract sentences before removing stopwords
        sentences = libs['sent_tokenize'](processed_text)
        
        if remove_stopwords:
            words = libs['word_tokenize'](processed_text)
            filtered_words = [word for word in words if word.lower() not in self.stop_words and len(word) > 2]
            processed_text = ' '.join(filtered_words)
        
        word_count = len(processed_text.split()) if processed_text else 0
        
        return {
            "processed_text": processed_text,
            "sentences": sentences,
            "word_count": word_count,
            "character_count": len(processed_text)
        }

class KeywordExtractor:
    def __init__(self):
        self.kw_model = kw_model
        self.nlp = nlp
    
    def extract_keywords_keybert(self, text, num_keywords=10, keyphrase_ngram_range=(1, 3)):
        """Extract keywords using KeyBERT with BERT embeddings"""
        if not text or len(text.strip()) < 50:
            return []
        
        try:
            # Configure KeyBERT parameters for optimal results
            keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=keyphrase_ngram_range,
                stop_words='english',
                top_k=num_keywords,
                use_maxsum=True,
                nr_candidates=num_keywords * 3,
                use_mmr=True,
                diversity=0.5
            )
            
            # Format results with confidence scores
            formatted_keywords = []
            for keyword, score in keywords:
                formatted_keywords.append({
                    'keyword': keyword,
                    'confidence': round(score, 3),
                    'length': len(keyword.split())
                })
            
            return formatted_keywords
            
        except Exception as e:
            st.error(f"KeyBERT extraction failed: {e}")
            return []
    
    def extract_keywords_tfidf(self, text, num_keywords=10):
        """Extract keywords using TF-IDF as fallback method"""
        if not text:
            return []
        
        try:
            # Configure TF-IDF vectorizer
            vectorizer = libs['TfidfVectorizer'](
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 3),
                max_df=0.85,
                min_df=2
            )
            
            # Handle single document by creating a small corpus
            sentences = libs['sent_tokenize'](text)
            if len(sentences) < 5:
                sentences = [p.strip() for p in text.split('\n') if p.strip()]
            
            if len(sentences) < 2:
                return []
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get mean TF-IDF scores
            mean_scores = tfidf_matrix.mean(axis=0).A1
            
            # Create keyword list with scores
            keyword_scores = list(zip(feature_names, mean_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Format results
            formatted_keywords = []
            for keyword, score in keyword_scores[:num_keywords]:
                formatted_keywords.append({
                    'keyword': keyword,
                    'confidence': round(score, 3),
                    'length': len(keyword.split())
                })
            
            return formatted_keywords
            
        except Exception as e:
            st.error(f"TF-IDF extraction failed: {e}")
            return []
    
    def extract_keywords(self, text, method='keybert', num_keywords=10):
        """Extract keywords using specified method with fallback"""
        
        if method == 'keybert':
            keywords = self.extract_keywords_keybert(text, num_keywords)
            if not keywords:  # Fallback to TF-IDF if KeyBERT fails
                keywords = self.extract_keywords_tfidf(text, num_keywords)
        else:
            keywords = self.extract_keywords_tfidf(text, num_keywords)
        
        return keywords

class NamedEntityExtractor:
    def __init__(self):
        self.nlp = nlp
        
        # Define entity categories of interest
        self.entity_categories = {
            'PERSON': 'People',
            'ORG': 'Organizations',
            'GPE': 'Geopolitical entities',
            'LOC': 'Locations',
            'DATE': 'Dates',
            'TIME': 'Times',
            'MONEY': 'Monetary values',
            'PERCENT': 'Percentages',
            'PRODUCT': 'Products',
            'EVENT': 'Events',
            'WORK_OF_ART': 'Works of art',
            'LAW': 'Laws and legal documents',
            'LANGUAGE': 'Languages'
        }
    
    def extract_entities(self, text, min_confidence=0.5):
        """Extract named entities with confidence filtering"""
        if not text:
            return {}
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Group entities by type
            entities_by_type = {}
            
            for ent in doc.ents:
                entity_type = ent.label_
                entity_text = ent.text.strip()
                
                # Filter out single characters and very short entities
                if len(entity_text) < 2:
                    continue
                
                # Initialize category if not exists
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = {}
                
                # Count entity occurrences
                if entity_text in entities_by_type[entity_type]:
                    entities_by_type[entity_type][entity_text]['count'] += 1
                else:
                    entities_by_type[entity_type][entity_text] = {
                        'count': 1,
                        'confidence': getattr(ent, 'confidence', 1.0),
                        'start_char': ent.start_char,
                        'end_char': ent.end_char
                    }
            
            # Format and filter results
            formatted_entities = {}
            for entity_type, entities in entities_by_type.items():
                if entity_type in self.entity_categories:
                    # Sort by count and take most frequent
                    sorted_entities = sorted(
                        entities.items(), 
                        key=lambda x: x[1]['count'], 
                        reverse=True
                    )
                    
                    formatted_entities[entity_type] = {
                        'category_name': self.entity_categories[entity_type],
                        'entities': []
                    }
                    
                    for entity_text, entity_info in sorted_entities[:10]:  # Top 10 per category
                        formatted_entities[entity_type]['entities'].append({
                            'text': entity_text,
                            'count': entity_info['count'],
                            'confidence': entity_info['confidence']
                        })
            
            return formatted_entities
            
        except Exception as e:
            st.error(f"Named entity extraction failed: {e}")
            return {}

class TextSummarizer:
    def __init__(self):
        self.summarizer = summarizer
        self.max_chunk_length = 1024
        self.min_summary_length = 50
        self.max_summary_length = 150
    
    def chunk_text(self, text, max_length=1000):
        """Split text into chunks suitable for summarization"""
        if not text:
            return []
        
        sentences = libs['sent_tokenize'](text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def summarize_text(self, text, summary_type='balanced'):
        """Generate summary using transformer model"""
        if not text or not self.summarizer:
            return ""
        
        try:
            # Adjust summary length based on type
            if summary_type == 'brief':
                max_length = 100
                min_length = 30
            elif summary_type == 'detailed':
                max_length = 250
                min_length = 100
            else:  # balanced
                max_length = 150
                min_length = 50
            
            # Handle long texts by chunking
            text_chunks = self.chunk_text(text, max_length=self.max_chunk_length)
            
            if not text_chunks:
                return ""
            
            summaries = []
            
            for i, chunk in enumerate(text_chunks[:3]):  # Limit to first 3 chunks
                try:
                    # Generate summary for chunk
                    summary = self.summarizer(
                        chunk,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False,
                        num_beams=4,
                        length_penalty=2.0,
                        early_stopping=True
                    )
                    
                    if summary and len(summary) > 0:
                        summaries.append(summary[0]['summary_text'])
                        
                except Exception as e:
                    st.warning(f"Failed to summarize chunk {i+1}: {e}")
                    continue
            
            if summaries:
                # Combine summaries if multiple chunks
                if len(summaries) == 1:
                    final_summary = summaries[0]
                else:
                    final_summary = ' '.join(summaries)
                
                return final_summary
            else:
                return ""
                
        except Exception as e:
            st.error(f"Summarization failed: {e}")
            return ""

class MetadataGenerator:
    def __init__(self):
        pass
    
    def generate_title(self, text, keywords, filename):
        """Generate an appropriate title for the document"""
        
        # Try to extract title from first few sentences
        sentences = libs['sent_tokenize'](text)
        
        # Look for title-like patterns in first few sentences
        for sentence in sentences[:3]:
            sentence = sentence.strip()
            # Check if sentence looks like a title
            if (len(sentence.split()) <= 15 and 
                len(sentence) <= 100 and 
                not sentence.endswith('.') and
                len(sentence) > 0 and
                sentence[0].isupper()):
                return sentence
        
        # Use top keywords to create a title
        if keywords and len(keywords) > 0:
            top_keywords = [kw['keyword'] for kw in keywords[:3]]
            generated_title = ' '.join(top_keywords).title()
            if len(generated_title) <= 100:
                return generated_title
        
        # Fallback to filename-based title
        title_from_filename = Path(filename).stem.replace('_', ' ').replace('-', ' ').title()
        return title_from_filename
    
    def calculate_quality_scores(self, extraction_metadata, text_length, entities_count, keywords_count):
        """Calculate quality and confidence scores"""
        
        scores = {
            'extraction_confidence': 0.0,
            'text_quality_score': 0.0,
            'completeness_score': 0.0
        }
        
        # Extraction confidence based on text length and source
        if text_length > 1000:
            scores['extraction_confidence'] += 0.4
        elif text_length > 500:
            scores['extraction_confidence'] += 0.3
        elif text_length > 100:
            scores['extraction_confidence'] += 0.2
        
        # Additional confidence from successful extraction
        if extraction_metadata.get('word_count', 0) > 0:
            scores['extraction_confidence'] += 0.3
        
        if 'title' in extraction_metadata:
            scores['extraction_confidence'] += 0.2
        
        scores['extraction_confidence'] = min(scores['extraction_confidence'], 1.0)
        
        # Text quality score
        if text_length > 500:
            scores['text_quality_score'] += 0.4
        
        word_count = extraction_metadata.get('word_count', 0)
        if word_count > 100:
            scores['text_quality_score'] += 0.3
        
        # Vocabulary diversity indicator
        if text_length > 0 and word_count > 0:
            char_to_word_ratio = text_length / word_count
            if 4 <= char_to_word_ratio <= 8:  # Reasonable character-to-word ratio
                scores['text_quality_score'] += 0.3
        
        scores['text_quality_score'] = min(scores['text_quality_score'], 1.0)
        
        # Completeness score based on extracted metadata richness
        components_found = 0
        total_components = 5
        
        if keywords_count > 0:
            components_found += 1
        if entities_count > 0:
            components_found += 1
        if text_length > 200:
            components_found += 1
        if extraction_metadata.get('title'):
            components_found += 1
        if word_count > 50:
            components_found += 1
        
        scores['completeness_score'] = components_found / total_components
        
        return scores
    
    def generate_comprehensive_metadata(self, text, extraction_metadata, filename):
        """Generate comprehensive structured metadata"""
        
        if not text:
            return self._create_empty_metadata(filename)
        
        # Extract semantic information
        keyword_extractor = KeywordExtractor()
        entity_extractor = NamedEntityExtractor()
        text_summarizer = TextSummarizer()
        
        keywords = keyword_extractor.extract_keywords(text, num_keywords=15)
        entities = entity_extractor.extract_entities(text)
        
        # Generate summary
        summary = text_summarizer.summarize_text(text)
        
        # Generate title
        title = self.generate_title(text, keywords, filename)
        
        # Calculate quality scores
        entities_count = sum(len(cat['entities']) for cat in entities.values())
        keywords_count = len(keywords)
        quality_scores = self.calculate_quality_scores(
            extraction_metadata, len(text), entities_count, keywords_count
        )
        
        # Structure metadata according to schema
        metadata = {
            'document_info': {
                'title': title,
                'filename': filename,
                'file_type': extraction_metadata.get('file_extension', 'unknown'),
                'file_size': extraction_metadata.get('file_size', 0),
                'creation_date': extraction_metadata.get('creation_time', ''),
                'processing_date': datetime.now().isoformat()
            },
            'content_analysis': {
                'language': 'en',  # Simplified for this implementation
                'word_count': len(text.split()) if text else 0,
                'character_count': len(text),
                'sentence_count': len(libs['sent_tokenize'](text)) if text else 0,
            },
            'semantic_metadata': {
                'keywords': [kw['keyword'] for kw in keywords],
                'keyword_details': keywords,
                'summary': summary,
            },
            'entities': self._structure_entities(entities),
            'quality_metrics': quality_scores,
            'extraction_metadata': extraction_metadata
        }
        
        return metadata
    
    def _structure_entities(self, entities):
        """Structure entities into organized categories"""
        structured = {
            'people': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'other_entities': {}
        }
        
        entity_mapping = {
            'PERSON': 'people',
            'ORG': 'organizations',
            'GPE': 'locations',
            'LOC': 'locations',
            'DATE': 'dates',
            'TIME': 'dates'
        }
        
        for entity_type, entity_data in entities.items():
            mapped_category = entity_mapping.get(entity_type, 'other_entities')
            
            if mapped_category in ['people', 'organizations', 'locations', 'dates']:
                structured[mapped_category].extend([
                    entity['text'] for entity in entity_data['entities']
                ])
            else:
                structured['other_entities'][entity_type] = [
                    entity['text'] for entity in entity_data['entities']
                ]
        
        return structured
    
    def _create_empty_metadata(self, filename):
        """Create empty metadata structure for failed extractions"""
        return {
            'document_info': {
                'title': Path(filename).stem,
                'filename': filename,
                'file_type': Path(filename).suffix,
                'file_size': 0,
                'creation_date': '',
                'processing_date': datetime.now().isoformat()
            },
            'content_analysis': {
                'language': 'unknown',
                'word_count': 0,
                'character_count': 0,
                'sentence_count': 0,
            },
            'semantic_metadata': {
                'keywords': [],
                'keyword_details': [],
                'summary': '',
            },
            'entities': {
                'people': [],
                'organizations': [],
                'locations': [],
                'dates': [],
                'other_entities': {}
            },
            'quality_metrics': {
                'extraction_confidence': 0.0,
                'text_quality_score': 0.0,
                'completeness_score': 0.0
            },
            'extraction_metadata': {}
        }

# Initialize extractors and processors
pdf_extractor = PDFTextExtractor()
docx_extractor = DOCXTextExtractor()
ocr_extractor = OCRTextExtractor()
text_preprocessor = TextPreprocessor()
metadata_generator = MetadataGenerator()

def extract_text_from_file(file_path, file_ext, filename):
    """Unified text extraction function for all supported file types"""
    
    if file_ext == '.pdf':
        text, metadata = pdf_extractor.extract_pdf_text(file_path)
    elif file_ext == '.docx':
        text, metadata = docx_extractor.extract_docx_text(file_path)
    elif file_ext == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            metadata = {'character_count': len(text), 'word_count': len(text.split())}
        except Exception as e:
            st.error(f"Text file reading failed: {e}")
            text, metadata = "", {}
    elif file_ext in ['.png', '.jpg', '.jpeg']:
        text, metadata = ocr_extractor.extract_ocr_text(file_path)
    else:
        st.error(f"Unsupported file type: {file_ext}")
        return "", {}
    
    # Add common metadata
    metadata['original_filename'] = filename
    metadata['file_extension'] = file_ext
    metadata['extraction_method'] = f"{file_ext[1:].upper()} processor"
    
    return text, metadata

def process_document_pipeline(file_path, filename, file_ext):
    """Complete document processing pipeline"""
    
    # Step 1: Extract text from document
    with st.status("Extracting text from document...", expanded=True) as status:
        st.write("🔄 Processing file...")
        text, extraction_metadata = extract_text_from_file(file_path, file_ext, filename)
        
        if not text:
            st.error("❌ No text extracted from document")
            return None
        
        st.write(f"✅ Extracted {len(text)} characters")
        
        # Step 2: Preprocess text
        st.write("🧹 Preprocessing text...")
        preprocessing_result = text_preprocessor.preprocess_text(text, normalize=True)
        processed_text = preprocessing_result['processed_text']
        st.write(f"✅ Processed {preprocessing_result['word_count']} words")
        
        # Step 3: Generate metadata
        st.write("🔬 Generating metadata...")
        metadata = metadata_generator.generate_comprehensive_metadata(
            processed_text, extraction_metadata, filename
        )
        st.write("✅ Metadata generation completed")
        
        status.update(label="✅ Processing completed!", state="complete", expanded=False)
    
    return metadata

def display_metadata_summary(metadata):
    """Display a comprehensive summary of generated metadata"""
    
    if not metadata:
        st.error("❌ No metadata to display")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Overview", "🏷️ Keywords & Entities", "📝 Summary", "📊 Quality Metrics"])
    
    with tab1:
        st.subheader("📄 Document Information")
        doc_info = metadata['document_info']
        content = metadata['content_analysis']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Title", doc_info['title'])
            st.metric("File Type", doc_info['file_type'])
            st.metric("Word Count", f"{content['word_count']:,}")
        
        with col2:
            st.metric("Language", content['language'].upper())
            st.metric("Character Count", f"{content['character_count']:,}")
            st.metric("Sentence Count", f"{content['sentence_count']:,}")
    
    with tab2:
        st.subheader("🏷️ Keywords")
        semantic = metadata['semantic_metadata']
        
        if semantic['keywords']:
            # Display keywords as tags
            keyword_html = ""
            for i, keyword in enumerate(semantic['keywords'][:15], 1):
                keyword_html += f'<span style="background-color: #e1f5fe; padding: 4px 8px; margin: 2px; border-radius: 12px; font-size: 0.9em;">{keyword}</span> '
            st.markdown(keyword_html, unsafe_allow_html=True)
        else:
            st.info("No keywords extracted")
        
        st.subheader("👥 Named Entities")
        entities = metadata['entities']
        
        entity_categories = [
            ('People', entities['people']),
            ('Organizations', entities['organizations']),
            ('Locations', entities['locations']),
            ('Dates', entities['dates'])
        ]
        
        for category, entity_list in entity_categories:
            if entity_list:
                st.write(f"**{category}:** {', '.join(entity_list[:5])}")
                if len(entity_list) > 5:
                    st.caption(f"... and {len(entity_list) - 5} more")
    
    with tab3:
        st.subheader("📝 Summary")
        if semantic['summary']:
            st.write(semantic['summary'])
        else:
            st.info("No summary generated")
    
    with tab4:
        st.subheader("📊 Quality Metrics")
        quality = metadata['quality_metrics']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Extraction Confidence", f"{quality['extraction_confidence']:.1%}")
        with col2:
            st.metric("Text Quality Score", f"{quality['text_quality_score']:.1%}")
        with col3:
            st.metric("Completeness Score", f"{quality['completeness_score']:.1%}")
        
        # Overall assessment
        overall_score = (quality['extraction_confidence'] + 
                        quality['text_quality_score'] + 
                        quality['completeness_score']) / 3
        
        if overall_score >= 0.8:
            assessment = "🌟 Excellent"
            color = "green"
        elif overall_score >= 0.6:
            assessment = "✅ Good"
            color = "blue"
        elif overall_score >= 0.4:
            assessment = "⚠️ Fair"
            color = "orange"
        else:
            assessment = "❌ Poor"
            color = "red"
        
        st.markdown(f"**Overall Assessment:** :{color}[{assessment} ({overall_score:.1%})]")

def create_metadata_visualization(metadata):
    """Create visualizations for metadata analysis"""
    
    if not metadata:
        return
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Document Metadata Analysis Dashboard', fontsize=14, fontweight='bold')
    
    # 1. Keywords confidence
    if 'keyword_details' in metadata['semantic_metadata']:
        keywords_data = metadata['semantic_metadata']['keyword_details'][:8]
        if keywords_data:
            keywords = [kw['keyword'][:20] for kw in keywords_data]  # Truncate long keywords
            confidences = [kw['confidence'] for kw in keywords_data]
            
            ax1.barh(range(len(keywords)), confidences, color='skyblue')
            ax1.set_yticks(range(len(keywords)))
            ax1.set_yticklabels(keywords, fontsize=8)
            ax1.set_xlabel('Confidence Score')
            ax1.set_title('Top Keywords by Confidence')
            ax1.invert_yaxis()
    
    # 2. Entity distribution
    entities = metadata['entities']
    entity_counts = {
        'People': len(entities['people']),
        'Organizations': len(entities['organizations']),
        'Locations': len(entities['locations']),
        'Dates': len(entities['dates']),
        'Other': sum(len(v) for v in entities['other_entities'].values())
    }
    
    entity_counts = {k: v for k, v in entity_counts.items() if v > 0}
    
    if entity_counts:
        ax2.pie(entity_counts.values(), labels=entity_counts.keys(), autopct='%1.1f%%')
        ax2.set_title('Named Entity Distribution')
    else:
        ax2.text(0.5, 0.5, 'No entities found', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Named Entity Distribution')
    
    # 3. Quality metrics
    quality = metadata['quality_metrics']
    metrics = ['Extraction\nConfidence', 'Text Quality\nScore', 'Completeness\nScore']
    scores = [quality['extraction_confidence'], quality['text_quality_score'], quality['completeness_score']]
    
    bars = ax3.bar(metrics, scores, color=['lightcoral', 'lightgreen', 'lightskyblue'])
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Score')
    ax3.set_title('Quality Metrics')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.2f}', ha='center', va='bottom')
    
    # 4. Content statistics
    content = metadata['content_analysis']
    stats_labels = ['Words', 'Sentences', 'Characters\n(hundreds)']
    stats_values = [
        content['word_count'],
        content['sentence_count'],
        content['character_count'] // 100
    ]
    
    ax4.bar(stats_labels, stats_values, color=['gold', 'orange', 'coral'])
    ax4.set_ylabel('Count')
    ax4.set_title('Content Statistics')
    
    # Add value labels
    for i, (label, value) in enumerate(zip(stats_labels, stats_values)):
        ax4.text(i, value + max(stats_values) * 0.01, f'{value:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

# Main Streamlit App
def main():
    st.title("📄 Automated Metadata Generation System")
    st.markdown("Upload a document and automatically extract comprehensive metadata using advanced NLP techniques.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("Supported formats: PDF, DOCX, TXT, PNG, JPG, JPEG")
        
        # Processing options
        st.subheader("Processing Options")
        keyword_method = st.selectbox("Keyword Extraction Method", ["keybert", "tfidf"], index=0)
        num_keywords = st.slider("Number of Keywords", 5, 20, 10)
        summary_type = st.selectbox("Summary Type", ["brief", "balanced", "detailed"], index=1)
    
    # File upload section
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg'],
        help="Upload a document to extract metadata"
    )
    
    if uploaded_file is not None:
        # Display file information
        file_details = {
            "Filename": uploaded_file.name,
            "File Size": f"{uploaded_file.size / 1024:.2f} KB",
            "File Type": uploaded_file.type
        }
        
        st.subheader("📋 File Information")
        for key, value in file_details.items():
            st.text(f"{key}: {value}")
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_button = st.button(
                "🚀 Process Document",
                type="primary",
                use_container_width=True
            )
        
        if process_button:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Process the document
                file_ext = Path(uploaded_file.name).suffix.lower()
                
                # Store processing time
                start_time = time.time()
                
                metadata = process_document_pipeline(tmp_file_path, uploaded_file.name, file_ext)
                
                processing_time = time.time() - start_time
                
                if metadata:
                    # Store results in session state
                    st.session_state['metadata'] = metadata
                    st.session_state['processing_time'] = processing_time
                    
                    st.success(f"✅ Processing completed in {processing_time:.2f} seconds!")
                    
                else:
                    st.error("❌ Failed to process document")
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
    
    # Display results if available
    if 'metadata' in st.session_state:
        st.header("📊 Processing Results")
        
        # Display metadata summary
        display_metadata_summary(st.session_state['metadata'])
        
        # Visualization
        st.subheader("📈 Metadata Visualization")
        fig = create_metadata_visualization(st.session_state['metadata'])
        if fig:
            st.pyplot(fig)
        
        # Export options
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            json_data = json.dumps(st.session_state['metadata'], indent=2, default=str)
            st.download_button(
                label="📄 Download JSON",
                data=json_data,
                file_name=f"metadata_{st.session_state['metadata']['document_info']['filename']}.json",
                mime="application/json"
            )
        
        with col2:
            # CSV export (flattened)
            def flatten_dict(d, prefix=''):
                flattened = {}
                for key, value in d.items():
                    new_key = f"{prefix}{key}" if prefix else key
                    if isinstance(value, dict):
                        flattened.update(flatten_dict(value, f"{new_key}_"))
                    elif isinstance(value, list):
                        if value and isinstance(value[0], str):
                            flattened[new_key] = '; '.join(value)
                        else:
                            flattened[new_key] = str(value)
                    else:
                        flattened[new_key] = str(value) if value is not None else ''
                return flattened
            
            flattened_data = flatten_dict(st.session_state['metadata'])
            df = pd.DataFrame([flattened_data])
            csv_data = df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download CSV",
                data=csv_data,
                file_name=f"metadata_{st.session_state['metadata']['document_info']['filename']}.csv",
                mime="text/csv"
            )
        
        # Processing statistics
        st.subheader("⏱️ Processing Statistics")
        processing_time = st.session_state.get('processing_time', 0)
        word_count = st.session_state['metadata']['content_analysis']['word_count']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Processing Time", f"{processing_time:.2f}s")
        with col2:
            st.metric("Words per Second", f"{word_count/processing_time:.0f}" if processing_time > 0 else "N/A")
        with col3:
            st.metric("Quality Score", f"{st.session_state['metadata']['quality_metrics']['completeness_score']:.1%}")

if __name__ == "__main__":
    main()
