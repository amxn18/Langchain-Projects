import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pdf2image import convert_from_path
import pytesseract

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    temperature=0.2,
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

st.set_page_config(
    page_title="ATS Resume Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("ATS Resume Analyzer")

def extractTextPypdf(pdfPath):
    loader = PyPDFLoader(pdfPath)
    docs = loader.load()
    return "\n".join(d.page_content for d in docs)

def extractTextOcr(pdfPath):
    images = convert_from_path(pdfPath)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

def extractResumeText(uploadedFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploadedFile.read())
        path = tmp.name

    text = extractTextPypdf(path)

    if len(text.strip()) < 300:
        text = extractTextOcr(path)

    os.remove(path)
    return text

def runLLM(prompt, resumeText, jdText):
    template = PromptTemplate(
        template=prompt,
        input_variables=["resume", "jd"]
    )

    chain = template | model | parser
    return chain.invoke({
        "resume": resumeText,
        "jd": jdText
    })

jobDescription = st.text_area(
    "Enter Job Description",
    height=200
)

resumeFile = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

col1, col2 = st.columns(2)

with col1:
    btn1 = st.button("Tell me about the Resume")
    btn2 = st.button("How can I improve my skills for this job?")

with col2:
    btn3 = st.button("What are the missing keywords?")
    btn4 = st.button("Percentage match with Job Description")

PROMPT1 = """
You are an experienced Technical HR Manager with over 15 years of expertise in talent acquisition and resume evaluation.

Your task is to thoroughly analyze the provided resume and deliver a comprehensive professional assessment.

Please provide a detailed analysis covering the following aspects:

1. **Candidate Summary**: Write a concise 3-4 sentence professional summary highlighting the candidate's overall profile, career level, and primary domain expertise.

2. **Technical Skills**: List all technical skills mentioned in the resume, categorizing them by proficiency level if indicated (e.g., programming languages, frameworks, tools, technologies).

3. **Professional Experience**: Summarize the candidate's work history including years of experience, key roles held, industries worked in, and notable achievements or projects.

4. **Educational Background**: Detail the candidate's educational qualifications including degrees, institutions, certifications, and any specialized training.

5. **Key Strengths**: Identify 4-5 major strengths based on the resume content such as unique skill combinations, leadership experience, domain expertise, or impressive accomplishments.

6. **Areas for Improvement**: Point out 3-4 potential weaknesses or gaps such as missing skills, lack of certain experiences, formatting issues, or areas that need better presentation.

7. **Overall Impression**: Provide your professional opinion on the candidate's marketability and potential fit for technical roles.

Resume:
{resume}

Please structure your response clearly with proper headings and bullet points for easy readability.
"""

PROMPT2 = """
You are a professional career development coach and mentor specializing in technology careers and skill development strategies.

Your objective is to provide personalized, actionable guidance to help the candidate bridge the gap between their current skills and the requirements of their target job.

Based on the resume and job description provided, please deliver a comprehensive skill improvement plan:

1. **Skill Gap Analysis**: Conduct a detailed comparison between the skills mentioned in the resume and those required in the job description. Clearly identify which required skills are missing or underdeveloped.

2. **Priority Skills to Develop**: List the top 5-7 most critical skills the candidate should focus on developing first, ranked by importance for the target role.

3. **Specific Learning Recommendations**: For each priority skill, suggest:
   - Specific online courses, certifications, or training programs (mention platforms like Coursera, Udemy, LinkedIn Learning, etc.)
   - Estimated time commitment required
   - Difficulty level and prerequisites

4. **Practical Experience Suggestions**: Recommend ways to gain hands-on experience such as:
   - Personal projects to build
   - Open-source contributions
   - Freelance opportunities
   - Volunteer work or internships

5. **Timeline and Roadmap**: Provide a realistic 3-6 month learning roadmap showing when and how to acquire each skill progressively.

6. **Additional Soft Skills**: Identify any soft skills or professional competencies that might be important for the role and suggest ways to develop them.

7. **Immediate Action Steps**: List 3-5 concrete actions the candidate can take this week to start their improvement journey.

Resume:
{resume}

Job Description:
{jd}

Please be specific, practical, and encouraging in your recommendations.
"""

PROMPT3 = """
You are an expert ATS (Applicant Tracking System) keyword optimization specialist with deep knowledge of how modern recruitment systems parse and rank resumes.

Your mission is to perform a thorough keyword analysis to help the candidate optimize their resume for maximum ATS compatibility and recruiter visibility.

Please provide a detailed keyword analysis structured as follows:

1. **Critical Keywords from Job Description**: Extract and list all important keywords, phrases, and requirements from the job description, organized by category:
   - Technical Skills (programming languages, frameworks, tools)
   - Soft Skills (leadership, communication, teamwork)
   - Qualifications (degrees, certifications, years of experience)
   - Job-Specific Terms (industry jargon, methodologies, processes)
   - Action Verbs and Competencies

2. **Keywords Present in Resume**: Identify which of the above critical keywords are already present in the candidate's resume.

3. **Missing High-Priority Keywords**: Highlight the most important keywords from the job description that are completely absent from the resume. Explain why each keyword is significant.

4. **Partially Matched Keywords**: List keywords where the resume has related terms but not exact matches (e.g., resume says "JavaScript" but job requires "React.js").

5. **Keyword Density Issues**: Identify any important keywords that appear in the resume but perhaps not frequently enough to be properly weighted by ATS systems.

6. **Recommended Keywords to Add**: Provide a prioritized list of 10-15 specific keywords and phrases the candidate should incorporate into their resume, along with suggestions on where to add them (summary, skills section, experience bullets).

7. **ATS Optimization Tips**: Offer 3-5 specific formatting or content tips to improve ATS parsing of the resume.

8. **Industry-Specific Terminology**: Suggest any relevant industry buzzwords or trending terms that could strengthen the resume for this particular role.

Resume:
{resume}

Job Description:
{jd}

Be thorough and specific with your keyword identification and recommendations.
"""

PROMPT4 = """
You are an advanced ATS (Applicant Tracking System) evaluation engine used by Fortune 500 companies to screen and rank candidate resumes against job requirements.

Your task is to perform a rigorous, data-driven assessment of how well the candidate's resume matches the job description, and provide a detailed scoring report.

Please conduct a comprehensive ATS evaluation following this structure:

1. **Overall Match Percentage**: Calculate and provide an overall match score between 0-100% based on:
   - Keyword alignment (40% weight)
   - Skills match (30% weight)
   - Experience relevance (20% weight)
   - Education and qualifications (10% weight)
   
   Show the breakdown of each component score.

2. **Detailed Skills Analysis**:
   - List all required skills from the job description
   - For each skill, indicate: ✓ Present, ✗ Missing, or ~ Partially Matched
   - Calculate skills match percentage

3. **Experience Alignment**:
   - Compare years of experience required vs. candidate's experience
   - Evaluate relevance of past roles to the target position
   - Assess industry experience match

4. **Education and Certifications**:
   - Check if educational requirements are met
   - Identify any missing certifications or qualifications
   - Note any additional credentials that add value

5. **Critical Missing Elements**: List the top 5-7 most important qualifications, skills, or experiences that the candidate lacks according to the job requirements.

6. **Competitive Strengths**: Highlight 3-5 strong points where the candidate exceeds or perfectly matches job requirements.

7. **ATS Ranking Prediction**: Indicate where this resume would likely rank:
   - Top 10% (Strong Match - Interview Likely)
   - Top 25% (Good Match - Interview Possible)
   - Top 50% (Moderate Match - Competitive Pool)
   - Below 50% (Weak Match - Unlikely to Advance)

8. **Hiring Recommendation**: Provide a clear final recommendation:
   - **Strong Yes**: Excellent match, highly recommend interview
   - **Yes**: Good match, recommend interview
   - **Maybe**: Moderate match, consider if in top candidate pool
   - **No**: Insufficient match for this role

9. **Improvement Priority**: If not a "Strong Yes", list the top 3 things the candidate must improve to become a strong contender.

Resume:
{resume}

Job Description:
{jd}

Be objective, thorough, and provide specific percentages and clear recommendations.
"""

if 'showResponse' not in st.session_state:
    st.session_state.showResponse = False
    st.session_state.responseContent = ""

if resumeFile and jobDescription.strip():

    if btn1 or btn2 or btn3 or btn4:
        st.session_state.showResponse = False
        with st.spinner("Please wait while we process your request..."):
            resumeText = extractResumeText(resumeFile)
            
            if btn1:
                with st.spinner("Analyzing resume..."):
                    result = runLLM(PROMPT1, resumeText, jobDescription)
                    st.session_state.responseContent = result
                    st.session_state.responseTitle = "Resume Overview"
                    st.session_state.showResponse = True
            
            elif btn2:
                with st.spinner("Generating skill improvement suggestions..."):
                    result = runLLM(PROMPT2, resumeText, jobDescription)
                    st.session_state.responseContent = result
                    st.session_state.responseTitle = "Skill Improvement Suggestions"
                    st.session_state.showResponse = True
            
            elif btn3:
                with st.spinner("🔎Identifying missing keywords..."):
                    result = runLLM(PROMPT3, resumeText, jobDescription)
                    st.session_state.responseContent = result
                    st.session_state.responseTitle = "Missing Keywords Analysis"
                    st.session_state.showResponse = True
            
            elif btn4:
                with st.spinner(" Calculating ATS match score..."):
                    result = runLLM(PROMPT4, resumeText, jobDescription)
                    st.session_state.responseContent = result
                    st.session_state.responseTitle = "ATS Match Result"
                    st.session_state.showResponse = True

    if st.session_state.showResponse:
        st.subheader(st.session_state.responseTitle)
        st.write(st.session_state.responseContent)

elif btn1 or btn2 or btn3 or btn4:
    st.error(" Please upload a resume and enter the job description.")