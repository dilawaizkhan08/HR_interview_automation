import os
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.requests import Request
import requests
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set the Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file.")

# Initialize FastAPI app
app = FastAPI()

# Initialize Jinja2Templates to point to the templates folder
templates = Jinja2Templates(directory="templates")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Route to serve the home page with the "Start Question Generation" button
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "message": "Welcome!"})

# Function to fetch raw text content from a Google Docs link
def fetch_raw_google_doc(file_url):
    """Fetch raw text content from a public Google Doc link."""
    try:
        if "docs.google.com/document" in file_url:
            doc_id = file_url.split('/d/')[1].split('/')[0]
            text_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"

            # Fetch the content
            response = requests.get(text_url)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
            return response.text
        else:
            return f"Unsupported file URL: {file_url}"
    except Exception as e:
        return f"Failed to fetch document content: {e}"

# Route to start question generation based on CSV
@app.post("/generate-questions")
async def generate_questions():
    try:
        # Step 1: Read the CSV file
        csv_path = "/Users/arslan/Desktop/Internship_Work/hr_interview_system/job-descriptions.csv"  # Path to your CSV file
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail="CSV file not found")
        
        df = pd.read_csv(csv_path)

        # Step 2: Extract job descriptions and CV links
        job_descriptions = df["JobDescription"].tolist()  # Assuming column is named 'JobDescription'
        cv_links = df["CV-Link"].tolist()  # Assuming column is named 'CV-Link'

        # Step 3: Generate questions using Groq for each job description and CV
        for job_desc, cv_link in zip(job_descriptions, cv_links):
            # Fetch CV content
            cv_content = fetch_raw_google_doc(cv_link)
            
            # Ensure that both job description and CV content were fetched successfully
            if cv_content.startswith("Failed"):
                raise HTTPException(status_code=404, detail=f"Failed to fetch CV from: {cv_link}")

            # Customize the prompt for Groq
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Job Description: {job_desc}\nCandidate's CV: {cv_content}\nGenerate 3 interview questions based on the job description and CV.",
                    }
                ],
                model="llama-3.3-70b-versatile",
            )

            # Print the generated questions to the terminal
            questions = chat_completion.choices[0].message.content
            print(f"Questions for Job Description: {job_desc}\n{questions}\n")

        return {"success": True, "message": "Questions generated successfully. Check the terminal for output."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the app using uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
