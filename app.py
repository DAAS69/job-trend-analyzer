from flask import Flask, render_template, request, jsonify
import PyPDF2
import requests
import os
from bs4 import BeautifulSoup
from docx import Document
from PIL import Image
import cv2
import numpy as np
import easyocr
import io
import re
from typing import Dict, List
import logging
from collections import Counter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_resume_data(file, file_type):
    text = ""
    
    if file_type == 'pdf':
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    
    elif file_type == 'docx':
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + '\n'
    
    elif file_type in ['png', 'jpg', 'jpeg']:
        # Convert uploaded file to numpy array
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])
        
        # Extract text from image
        results = reader.readtext(img)
        text = ' '.join([result[1] for result in results])
        
        # Reset file pointer for potential reuse
        file.seek(0)

    # Enhanced parsing logic
    def clean_text(text: str) -> str:
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    def find_skills(text: str) -> List[str]:
        # Common skill patterns
        skill_patterns = {
            'languages': r'(?:python|java|javascript|typescript|c\+\+|ruby|php|swift|kotlin|go|rust)\b',
            'web_tech': r'(?:html5?|css3?|react\.?js|angular|vue\.?js|node\.?js|express\.?js|django|flask|spring)\b',
            'databases': r'(?:sql|mysql|postgresql|mongodb|oracle|sqlite|redis|elasticsearch)\b',
            'cloud': r'(?:aws|amazon|azure|gcp|google cloud|cloud computing|docker|kubernetes|k8s)\b',
            'ml_ai': r'(?:machine learning|artificial intelligence|deep learning|neural networks|nlp|computer vision|tensorflow|pytorch)\b',
            'tools': r'(?:git|jenkins|travis|ci/cd|jira|agile|scrum|docker|kubernetes)\b'
        }
        
        found_skills = set()
        clean_txt = clean_text(text)
        
        # Look for skill patterns
        for category, pattern in skill_patterns.items():
            matches = re.finditer(pattern, clean_txt)
            for match in matches:
                # Get some context around the skill (helps verify it's actually a skill mention)
                start = max(0, match.start() - 50)
                end = min(len(clean_txt), match.end() + 50)
                context = clean_txt[start:end]
                
                # Only add if it appears in a skills-related context
                if any(marker in context for marker in ['skills', 'technologies', 'programming', 'developed', 'built', 'implemented']):
                    found_skills.add(match.group())
        
        return list(found_skills)
    
    def find_education(text: str) -> List[Dict[str, str]]:
        education_list = []
        
        # Education patterns
        degree_pattern = r'(?:bachelor|master|phd|b\.?(?:tech|e|sc|a)|m\.?(?:tech|e|sc|a)|doctorate)'
        univ_pattern = r'(?:university|college|institute|school)'
        year_pattern = r'(?:19|20)\d{2}'
        
        # Find education blocks
        edu_blocks = re.finditer(
            rf'({degree_pattern}[\s\w]*?)(?:{univ_pattern}[\s\w]*?)(?:{year_pattern})?',
            text.lower(),
            re.IGNORECASE
        )
        
        for match in edu_blocks:
            education_info = {
                'degree': match.group(1).strip(),
                'full_text': match.group(0).strip()
            }
            # Try to extract year if present
            year_match = re.search(year_pattern, match.group(0))
            if year_match:
                education_info['year'] = year_match.group(0)
            
            education_list.append(education_info)
        
        return education_list

    # Process the text
    cleaned_text = clean_text(text)
    skills = find_skills(cleaned_text)
    education = find_education(text)  # Use original text for better matching

    return {
        'skills': skills,
        'education': education
    }

def search_linkedin_jobs(skills):
    jobs = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    # Combine related skills for better job matching
    skill_groups = []
    for skill in skills:
        related_skills = [s for s in skills if are_skills_related(skill, s)]
        if related_skills not in skill_groups:
            skill_groups.append(related_skills)
    
    for skill_group in skill_groups:
        search_query = ' '.join(skill_group[:2])
        url = f"https://www.linkedin.com/jobs/search?keywords={search_query}"
        
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            job_cards = soup.find_all('div', {'class': 'base-card'})
            
            for card in job_cards[:3]:
                title_elem = card.find('h3', {'class': 'base-search-card__title'})
                company_elem = card.find('h4', {'class': 'base-search-card__subtitle'})
                location_elem = card.find('span', {'class': 'job-search-card__location'})
                link_elem = card.find('a', {'class': 'base-card__full-link'})
                
                if title_elem and company_elem and link_elem:
                    job_url = link_elem.get('href', '').split('?')[0]  # Remove tracking parameters
                    jobs.append({
                        'title': title_elem.text.strip(),
                        'company': company_elem.text.strip(),
                        'location': location_elem.text.strip() if location_elem else 'N/A',
                        'url': job_url,
                        'skills': skill_group
                    })
        except Exception as e:
            print(f"Error fetching jobs for {search_query}: {str(e)}")
            continue
    
    return jobs

def are_skills_related(skill1: str, skill2: str) -> bool:
    # Define related skill groups
    skill_groups = [
        {'javascript', 'react', 'html', 'css', 'nodejs'},
        {'python', 'django', 'flask', 'machine learning', 'ai'},
        {'java', 'spring', 'hibernate'},
        {'docker', 'kubernetes', 'aws', 'devops'},
    ]
    
    for group in skill_groups:
        if skill1.lower() in group and skill2.lower() in group:
            return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Get file extension
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    allowed_extensions = {
        'pdf': 'pdf',
        'docx': 'docx',
        'doc': 'docx',
        'png': 'png',
        'jpg': 'jpg',
        'jpeg': 'jpg'
    }
    
    if file_ext not in allowed_extensions:
        return jsonify({'error': 'Invalid file format. Allowed formats: PDF, DOCX, DOC, PNG, JPG'})
    
    try:
        resume_data = extract_resume_data(file, allowed_extensions[file_ext])
        jobs = search_linkedin_jobs(resume_data['skills'])
        return jsonify({
            'resume_data': resume_data,
            'jobs': jobs
        })
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'})

@app.route('/api/trends')
def get_trends():
    """Get real-time job market trends"""
    try:
        logger.info("Fetching job market trends")
        
        # Fetch job postings
        skills_to_track = ['python', 'javascript', 'java', 'react', 'aws', 'docker']
        market_data = {
            'skills': {},
            'companies': {},
            'locations': {}
        }
        
        for skill in skills_to_track:
            jobs = search_linkedin_jobs([skill])
            if jobs:
                # Count jobs by skill
                market_data['skills'][skill] = len(jobs)
                
                # Aggregate company data
                for job in jobs:
                    company = job['company']
                    market_data['companies'][company] = market_data['companies'].get(company, 0) + 1
                    
                    location = job['location']
                    market_data['locations'][location] = market_data['locations'].get(location, 0) + 1

        # Sort and limit to top 5
        market_data['companies'] = dict(sorted(market_data['companies'].items(), 
                                             key=lambda x: x[1], 
                                             reverse=True)[:5])
        market_data['locations'] = dict(sorted(market_data['locations'].items(), 
                                             key=lambda x: x[1], 
                                             reverse=True)[:5])

        logger.info(f"Found data for {len(market_data['skills'])} skills")
        return jsonify({'market_data': market_data})

    except Exception as e:
        logger.error(f"Error fetching trends: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search')
def search():
    try:
        query = request.args.get('q', '').lower()
        logger.info(f"Searching for: {query}")
        
        if not query:
            return jsonify({
                'results': {
                    'jobs': [],
                    'skills': [],
                    'locations': []
                }
            })

        # Search for jobs using the query
        jobs = search_linkedin_jobs([query])
        
        # Extract unique skills from found jobs
        skills = set()
        locations = set()
        
        for job in jobs:
            skills.update(job.get('skills', []))
            if job.get('location'):
                locations.add(job['location'])
        
        return jsonify({
            'results': {
                'jobs': jobs,
                'skills': list(skills),
                'locations': list(locations)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return jsonify({
            'error': 'Search failed',
            'results': {
                'jobs': [],
                'skills': [],
                'locations': []
            }
        }), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
