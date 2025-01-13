                                                 #JOB MATCHING AND INTERVIEW ANALYSIS TOOL



#importing dependency
from typing import Annotated
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
import os
from llama_index.core import SimpleDirectoryReader
import chromadb
from langchain_community.llms import Ollama
from fastapi import File, UploadFile
import assemblyai as aai
import json

#setup the system prompt for model
system_prompt="""
You are an assistant. Your goal help and assist in the job hiring process.by Providing analytics to show which candidate is a good fit for a given job,
        highlighting the percentage of required skills that candidate possesses
"""
#initilizing llama2 model
llm = Ollama(model="llama2",base_url  = "http://localhost:11434",system=system_prompt,temperature=0.0)  

#setup assemblyai
aai.settings.api_key = "35a89577b4804bbf9d1576442bb7621c"

#initilizing FastAPI
app = FastAPI()

#initilizing chromadb
chroma_client = chromadb.PersistentClient(path='apidata')
resume_collection = chroma_client.get_or_create_collection(name="candidates_resume")
job_collection = chroma_client.get_or_create_collection(name="job_details")

#function for interview
def analyse_interview(input):
    try:
        transcriber = aai.Transcriber()
        interview = transcriber.transcribe(input)
    
    except:
        print("failed")
    
    prompt1=f"""<|USER|>analyze the interview conversation of candidate and give information like what is interview about, how confident the candidate is, how accurate the candidate answers are etc.
      here is the interview conversation= {interview.text}
       
       <|ASSISTANT|>"""
    
    return llm.invoke(prompt1)

#index page
@app.get("/")
async def read_root():
    return {"Hello": "Dhiwise"}

#recurter page: where recurters can upload the job description 
@app.post("/recurter/")
async def create_files(files: Annotated[list[bytes], File(description="Multiple files as bytes")],):
    job_details=[]
    ids=[]
    count=0
    for file in files:
        data=file.decode("utf-8")
        job_details.append(data)
        ids.append(str(count))
        with open (f"job_details{count}.json", "w") as f:
            json.dump(data  , f)
            count+=1
    #pushing the job discription data  to vector database    
    job_collection.upsert(documents=job_details,ids=ids)
    
    return {"file_sizes": [len(file) for file in files]}

#candidate page: where candidates can upload the resumes 
@app.post("/candidates/")
async def create_upload_files(files: Annotated[list[UploadFile], File(description="Multiple files as UploadFile")],):
    for file in files:
        data = await file.read()
        with open(file.filename, "wb") as f:
            f.write(data)
        os.replace(f"D:\\Beginner\\dhiwise\\{file.filename}",f"D:\\Beginner\\dhiwise\\krati\\resumes\\{file.filename}")
    candidate_data = SimpleDirectoryReader("D:/Beginner/dhiwise/krati/resumes").load_data()
    #data preprocessing
    documents = []
    metadata = []
    ids = []
    for items in candidate_data:
        items=dict(items)
        documents.append(items["text"])
        metadata.append(items["metadata"])
        ids.append(items["id_"])
    #pushing the resume data to vector database
    resume_collection.upsert(documents=documents,metadatas=metadata,ids=ids)
    return {"filenames": [file.filename for file in files]}

#output page: compare the resume and job description
@app.get('/output')
async def get_prompt(prompt : str):
    job_result = job_collection.query(query_texts=[prompt],n_results=2)#fecthing the job discription from vectorDB
    candidate_result = resume_collection.query(query_texts=[prompt],n_results=2)#fectching the resumes from vectorDB
    new_prompt = prompt + str(job_result["documents"]) + str(candidate_result["documents"][0])#passing the data with prompt
    return  llm.invoke(new_prompt)#generating the result

#interview analysis output
@app.post("/interview_analysis")
async def create_file(file: Annotated[bytes, File()]):
    analyzed = analyse_interview(file)
    return analyzed #returning interview analysis response

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)