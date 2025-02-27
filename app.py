import os
import json
import streamlit as st
import PyPDF2
from openai import OpenAI

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file object."""
    text = ""
    try:
        # Use PyPDF2 to read the uploaded PDF file
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

def analyze_resume_relevancy(resume_text, job_description):
    """
    Call OpenAI GPT-o1 to analyze resume relevancy for the job description.
    This version also evaluates the similarity of soft skills and technical skills.
    """
    # Initialize OpenAI client using your API key from environment variables
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create the prompt containing both the job description and resume text
    prompt = f"""
    Extract the relevancy of this resume to this job description:
    
    JOB DESCRIPTION:
    {job_description}
    
    RESUME:
    {resume_text}
    
    Your output should contain a JSON with {{
        "relevancy_score": <float value between 0 and 100 indicating overall relevancy>,
        "soft_skills_similarity": <float value between 0 and 100 indicating soft skills similarity>,
        "technical_skills_similarity": <float value between 0 and 100 indicating technical skills similarity>,
        "highlights": <array of short string highlights (each less than 20 characters) relevant to the job description>,
        "reasons": <string providing reasons for the scores>,
        "skill_gaps": [
            {{"name": "skill_name", "value": <float between 0 and 1>}}
            ...
        ]
    }}
    """
    
    # Call the GPT-o1 model with a lower temperature for more consistent output
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant that analyzes resume relevancy for job descriptions. Provide honest, accurate assessments in JSON format."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    # Parse the JSON output from the response
    result = json.loads(response.choices[0].message.content)
    return result

def main():
    st.title("Resume vs. Job Description Similarity Checker")
    st.write("Upload a resume PDF and enter a job description to see an analysis of relevancy, including soft skills and technical skills similarity.")

    # Text area for job description (pre-filled with an example)
    job_description = st.text_area("Enter Job Description:", value="""HeyMilo AI is one of the fastest-growing companies in the recruiting tech space. We leverage AI to simplify and enhance the hiring process, helping recruiters focus on what matters most—finding the right talent. We aim to create impeccable user experiences for both candidates and employers.

The Role:
We're looking for a skilled UX Designer to join us part-time. As part of our team, you'll play a key role in redesigning and refining our platform to align with user needs and support our growth. You'll work closely with product managers, engineers, and other stakeholders to translate user requirements into clear, functional designs. From lo-fi wireframes to high-fidelity prototypes in Figma, you'll create designs that are practical, intuitive, and aligned with the demands of our rapidly growing user base.

Responsibilities:
- Work with the product and engineering teams to identify and prioritize user needs.
- Develop wireframes, prototypes, and high-fidelity designs for the platform using Figma.
- Ensure designs are functional, user-focused, and scalable as our platform grows.
- Use feedback from stakeholders and users to refine designs and improve usability.
- Collaborate with cross-functional teams to ensure seamless implementation of design solutions.

Requirements:
- Ability to move fast. We ideate on Monday and ship by Friday.
- Strong experience in UX/UI design, particularly for web platforms.
- Expertise in Figma and a solid understanding of responsive design.
- Ability to turn complex requirements into user-friendly designs.
- Experience working in fast-paced environments with evolving priorities.
- Bonus: Familiarity with AI-driven platforms or tools.
- Bonus: Bachelor's degree in Design or a related field, or equivalent experience.

Your work will directly impact how we serve our users and scale our platform. This is an opportunity to shape the user experience of a product used by thousands around the world.
""")
    
    # File uploader for the resume PDF
    uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
    
    if uploaded_file and job_description:
        resume_text = extract_text_from_pdf(uploaded_file)
        
        if resume_text:
            st.subheader("Extracted Resume Text")
            st.text(resume_text)
            
            if st.button("Analyze Resume Relevancy"):
                try:
                    result = analyze_resume_relevancy(resume_text, job_description)
                    
                    st.header("Resume Relevancy Analysis")
                    st.write(f"**Overall Relevancy Score:** {result['relevancy_score']}/100")
                    st.write(f"**Soft Skills Similarity:** {result['soft_skills_similarity']}/100")
                    st.write(f"**Technical Skills Similarity:** {result['technical_skills_similarity']}/100")
                    
                    st.subheader("Highlights")
                    for highlight in result['highlights']:
                        st.write(f"- {highlight}")
                    
                    st.subheader("Reasons")
                    st.write(result['reasons'])
                    
                    st.subheader("Skill Gaps")
                    for gap in result['skill_gaps']:
                        st.write(f"- {gap['name']}: {gap['value']} (0 = minor gap, 1 = major gap)")
                    
                    # Provide an option to download the analysis as a JSON file
                    json_result = json.dumps(result, indent=2)
                    st.download_button("Download Analysis JSON", data=json_result, file_name="resume_analysis_result.json")
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
        else:
            st.error("Failed to extract text from the PDF. Please try a different file.")

if __name__ == "__main__":
    main()
