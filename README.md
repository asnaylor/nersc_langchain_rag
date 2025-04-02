# NERSC LangChain RAG 

## Overview
Retrieval-Augmented Generation (RAG) is a technique that combines the strengths of retrieval-based and generation-based models to enhance the quality of generated content. This Jupyter notebook demo will walk you through the steps to set up and run a RAG model using NERSC's supercompters. For more explanation about RAG, visit this [blog](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/).

## Prerequisites
1. **NERSC Account**: Ensure you have an active NERSC account with sufficient "Node Hours" to execute GPU compute.
2. **Hugging Face Access Token**: For models that require special access, create a Hugging Face account if you don't already have one. Follow [these instructions](https://huggingface.co/docs/hub/en/security-tokens) to generate a user access token and update the `HF_TOKEN` variable with your token.
3. **Nvidia NGC API Key**: To access specific models and containers, create an Nvidia Developer account if you don't have one. Follow [these instructions](https://docs.nvidia.com/ai-enterprise/deployment/spark-rapids-accelerator/latest/appendix-ngc.html) to generate an NGC API key and update the `NGC_API_KEY` variable with your key.

## Installation
To get started on a NERSC supercomputer, follow these steps to clone the repository, install the required dependencies, and pull the necessary containers:

1. Clone the Repository:
```bash
git clone https://github.com/yourusername/nersc_langchain_rag.git
cd nersc_langchain_rag
```

2. Create a Conda Environment and Jupyter Kernel:
```bash
ENV_DIR=$SCRATCH/langchain
module load conda
mamba create --prefix ${ENV_DIR} python=3.12 ipykernel -y
mamba activate ${ENV_DIR}
python -m ipykernel install \
    --user --name langchain --display-name LangChain
pip install langchain langchain-openai langchain-community langchain-nvidia-ai-endpoints pypdf
mamba install faiss-gpu -y
```

3. Pull the containers:
```bash
#The vLLM image is already on shifter - vllm/vllm-openai:v0.7.3

#Provide podman with NGC_API_KEY
echo $NGC_API_KEY | podman-hpc login nvcr.io --username '$oauthtoken' --password-stdin

#Download embedding and ranking containers from NGC
podman-hpc pull nvcr.io/nim/nvidia/nv-embedqa-e5-v5:1.5.0
podman-hpc pull nvcr.io/nim/nvidia/nv-rerankqa-mistral-4b-v3:1.0.2
```

## Usage

Once you have completed the setup, you can start running the Jupyter Notebook on [NERSC JupyterHub](https://jupyter.nersc.gov/) to execute the RAG model (select the "LangChain" Python kernel). The notebook contains instructions and code snippets to help you understand the process.


# Notebook


## Overview
This RAG example will be creating a sophisticated question-answering (Q&A) chatbot focused on the [NERSC 10-Year strategic plan](https://www.nersc.gov/about/nersc-10-year-strategic-plan/). This document outlines NERSC’s mission, goals, science drivers, planned initiatives, and technology strategy for the coming decade. By combining advanced language models and retrieval techniques, the chatbot will be able to provide accurate and contextually relevant answers.

The chatbot chain is constructed using the LangChain library and integrates the following models:
- **Meta Llama 3.3 70B Instruct Model**: Served via vLLM, this model provides the foundational language generation capabilities.
- **NVIDIA Retrieval QA E5 Embedding Model**: Utilized for embedding-based retrieval, this model is served by NVIDIA NIM.
- **NVIDIA Retrieval QA Mistral 4B Reranking Model**: Employed for reranking retrieved results, this model is also served by NVIDIA NIM.

These models work together to enhance the chatbot's ability to understand and generate responses based on the content of the NERSC 10-Year strategic plan. The embedding model helps in retrieving relevant chunks of text, while the reranking model ensures that the most relevant information is prioritized.


## LLM

In this section, we will deploy the foundational language model using the `deploy_llm` command. This command will run the model in the background on NERSC's GPU compute resources.



```python
from utils import allocate_gpu_resources, generate_api_key

# Generate an API key for the LLM service
llm_api_key = generate_api_key()

# Command to deploy the LLM using vLLM
llm_command = (
    "srun -n 1 --cpus-per-task=128 --gpus-per-task=4 "
    "  shifter "
    "    --image=vllm/vllm-openai:v0.7.3 "
    "    --module=gpu,nccl-plugin "
    "    --env=HF_TOKEN=\"$(cat $HOME/.hf_token)\" "
    "    --env=HF_HOME=\"$SCRATCH/huggingface/\" "
    "        vllm serve meta-llama/Llama-3.3-70B-Instruct "
    f"             --api-key {llm_api_key} --tensor-parallel-size 4"
)

# Allocate GPU resources via Slurm and start the LLM process
llm_process = allocate_gpu_resources(
    account="dasrepo",
    num_gpus=4,
    queue="interactive",
    time="01:00:00",
    job_name="vLLM_RAG",
    commands=llm_command
)

```

    Started process with PID: 232266


Check the job has started and once started get vLLM address and check the model has been loaded.


```python
!squeue --me
```

                 JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
              37406791 urgent_gp vLLM_RAG asnaylor  R       0:03      1 nid200392



```python
from utils import get_node_address, monitor_service_status

# Get the vLLM address
vLLM_address=f"http://{get_node_address('vLLM_RAG')}.chn.perlmutter.nersc.gov:8000/v1"

# Check if the service is up and the model is loaded
monitor_service_status(vLLM_address, endpoint="/models", api_key=llm_api_key)
```

    An error occurred while checking the service status: HTTPConnectionPool(host='nid200392.chn.perlmutter.nersc.gov', port=8000): Max retries exceeded with url: /v1/models (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f3963bfc770>: Failed to establish a new connection: [Errno 111] Connection refused'))
    Service is not up yet. Checking again in 60 seconds...
    An error occurred while checking the service status: HTTPConnectionPool(host='nid200392.chn.perlmutter.nersc.gov', port=8000): Max retries exceeded with url: /v1/models (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f396289e8a0>: Failed to establish a new connection: [Errno 111] Connection refused'))
    Service is not up yet. Checking again in 60 seconds...
    An error occurred while checking the service status: HTTPConnectionPool(host='nid200392.chn.perlmutter.nersc.gov', port=8000): Max retries exceeded with url: /v1/models (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x7f396289f0e0>: Failed to establish a new connection: [Errno 111] Connection refused'))
    Service is not up yet. Checking again in 60 seconds...
    Service is up.


## Embedding + Reranking
In this section, we will deploy the embedding and reranking models. These models will be used to retrieve and rank relevant information from the document.



```python
# Pre-command to set up the environment and cache directory
pre_command = "mkdir -p ${SCRATCH}/nim_cache; export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman; "

# Command to deploy the embedding model
embed_command = (
    "(srun -n 1 --cpus-per-task=32 --gpus-per-task=1 --overlap"
    "  podman-hpc run --rm -it --gpu --userns keep-id:uid=1000,gid=1000 "
    "    --volume \"${SCRATCH}/nim_cache:/opt/nim/.cache\" "
    "    --env=NGC_API_KEY=\"$(grep apikey ~/.ngc/config | awk '{printf $3}')\" "
    "    --env=NIM_HTTP_API_PORT=\"8000\" "
    "    -p \"8010:8000\" "
    "        nvcr.io/nim/nvidia/nv-embedqa-e5-v5:1.5.0 ) & "
)

# Command to deploy the reranking model
rerank_command = (
    "(srun -n 1 --cpus-per-task=32 --gpus-per-task=1 --overlap"
    "  podman-hpc run --rm -it --gpu --userns keep-id:uid=1000,gid=1000 "
    "    --volume \"${SCRATCH}/nim_cache:/opt/nim/.cache\""
    "    --env=NGC_API_KEY=\"$(grep apikey ~/.ngc/config | awk '{printf $3}')\" "
    "    --env=NIM_HTTP_API_PORT=\"8000\""
    "    -p \"8020:8000\" "
    "        nvcr.io/nim/nvidia/nv-rerankqa-mistral-4b-v3:1.0.2 ) & "
)

# Combine the commands and wait for both processes to complete
embed_rerank_command = pre_command + embed_command + rerank_command + "wait"

# Allocate GPU resources via Slurm and start the embedding and reranking processes
embed_rerank_process = allocate_gpu_resources(
    account="dasrepo",
    num_gpus=2,
    queue="shared_interactive",
    time="01:00:00",
    job_name="embed_rerank_RAG",
    commands=embed_rerank_command
)

```

    Started process with PID: 233004


Check the job has started and once started get addresses and check the models have been loaded.


```python
!squeue --me
```

                 JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
              37406794 shared_ur embed_re asnaylor  R       0:12      1 nid200325
              37406791 urgent_gp vLLM_RAG asnaylor  R       0:24      1 nid200392



```python
# Get the addresses for embedding and reranking services
embed_rerank_address=f"http://{get_node_address('embed_rerank_RAG')}.chn.perlmutter.nersc.gov"
embed_address=f"{embed_rerank_address}:8010/v1"
rerank_address=f"{embed_rerank_address}:8020/v1"
```


```python
# Check if the embedding service is up
monitor_service_status(embed_address, endpoint="/health/ready")
```

    Service is up.



```python
# Check if the reranking service is up
monitor_service_status(rerank_address, endpoint="/health/ready")
```

    Service is up.


## Connect to services
In this section, we will connect to the deployed services using the LangChain library. This will involve initializing the language model, embedding model, and reranking model.



```python
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank

# Initialize the language model with the vLLM address and API key
llm = ChatOpenAI(
    model="meta-llama/Llama-3.3-70B-Instruct",
    base_url=vLLM_address,
    api_key=llm_api_key
)

# Initialize the embedding model with the embedding service address
embedder = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5",
    base_url=embed_address
)

# Initialize the reranking model with the reranking service address
reranker = NVIDIARerank(
    model="nvidia/nv-rerankqa-mistral-4b-v3",
    base_url=rerank_address
)
```

## Process Document
In this section, we will download and process the NERSC 10-Year strategic plan PDF document. This involves downloading the document, splitting it into manageable chunks, and embedding the chunks for retrieval.



```python
from utils import download_pdf

# URL and local path for the NERSC 10-Year strategic plan PDF
nersc_10yr_plan_url = 'https://www.nersc.gov/assets/Annual-Reports/2024-2034-NERSC-10-yr-Strategic-Plan-v2.pdf'
nersc_10yr_plan_pdf = 'nersc_10yr_plan.pdf'

# Download the PDF document
download_pdf(nersc_10yr_plan_url, nersc_10yr_plan_pdf)
```

    File already exists at nersc_10yr_plan.pdf. Skipping download.



```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " ", ""],
)

# Load the PDF document
loader = PyPDFLoader(nersc_10yr_plan_pdf)
document = loader.load()

# Split the document into smaller chunks
document_chunks = text_splitter.split_documents(document)
```


```python
print("Number of pages: ", len(document))
print("Number of chunks from the document:", len(document_chunks))
```

    Number of pages:  28
    Number of chunks from the document: 149


Embed the document chunks using the embedding model.


```python
from langchain_community.vectorstores import FAISS
import os
import pickle

# Embed the document chunks
vectorstore = FAISS.from_documents(document_chunks, embedder)

# Save the vector store to a file
vector_store_path = f"{os.getenv('SCRATCH')}/rag_langchain_db_vectorstore.pkl"
with open(vector_store_path, "wb") as f:
    pickle.dump(vectorstore, f)
```

Set up the retriever with contextual compression.


```python
from langchain.retrievers import ContextualCompressionRetriever

# Initialize the retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Initialize the compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=retriever
)
```

## Setup chain

In this section, we will set up the chain to process the retrieved documents and generate answers. This involves defining the prompt template and creating the QA chain.


```python
from langchain_core.prompts import ChatPromptTemplate

# Define the prompt template for the chatbot
messages = [
    {
        "role": "system",
        "content": (
            "You are an advanced AI assistant with expertise in extracting and summarizing information from documents. "
            "Your task is to answer questions based on the content of the provided PDF document. "
            "Please ensure your responses are accurate, concise, and directly related to the content of the PDF."
        )
    },
    {
        "role": "user",
        "content": "Here is the content of the PDF document you will use to answer questions:\n\n{pdf_content}"
    },
    {
        "role": "user",
        "content": (
            "Based on the content above, please provide a detailed answer to the following question. "
            "Make sure your answer is comprehensive and references specific information from the PDF content when necessary."
        )
    },
    {
        "role": "user",
        "content": "Question: {question}"
    }
]

# Create the chat prompt template
rag_chat_prompt = ChatPromptTemplate.from_messages(messages)
```

Define the chain to process the documents and generate answers.


```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Convert loaded documents into strings by concatenating their content and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the QA chain
qa_chain = (
    {"pdf_content": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | rag_chat_prompt
    | llm
    | StrOutputParser()
)
```

## Q&A with retrieval
In this section, we will use the QA chain to answer questions based on the NERSC 10-Year strategic plan. This involves invoking the QA chain with a question and printing the answer.



```python
answer = qa_chain.invoke(
    "Write me a short bullet point summary of NERSC's AI goals for the next 10 years"
)

print(answer)
```

    Based on the provided PDF content, here is a short bullet point summary of NERSC's AI goals for the next 10 years:
    
    * Determine AI capability requirements for NERSC science and HPC system operations within 1-2 years
    * Produce initial design and prototypes of components for a next-generation NERSC AI services platform within 1-2 years
    * Seamlessly integrate AI into all science workflows, opening up new potential for scientific discovery, by exploiting innovations in other areas of NERSC's strategy
    * Leverage AI in automated monitoring, including creating tools and dashboards to prepare for automated monitoring, and deploying automated monitoring on the NERSC-10 system within 3-5 years
    * Develop system hardware and software that liberates scientists to apply large AI models, including accelerators for pervasive AI, workflow, and data management
    
    Note: These goals are part of NERSC's broader strategy to build a pervasive AI ecosystem and achieve a self-driving smart facility within the next 10 years.



```python
answer = qa_chain.invoke(
    "Based on the NERSC strategic AI goals, create a simple bullet point implentation plan for those goals"
)

print(answer)
```

    Based on the NERSC Strategic Plan FY2024-2032, here is a simple bullet point implementation plan for the AI goals:
    
    * **Short-term (1-2 years)**:
      * Host foundational AI models and datasets
      * Develop intelligent AI-driven interfaces to compute
      * Establish a Cross-Team Center Roadmap Committee to review, schedule, and plan AI-related projects
      * Develop a plan for center-wide monitoring, including milestones, software roadmap, hardware architecture, cost, and FTE count, in the context of NERSC-10
    * **Medium-term (3-5 years)**:
      * Create tools and dashboards to prepare for automated monitoring, including leveraging AI
      * Deploy automated monitoring on the NERSC-10 system
      * Develop a Center Roadmap Tool to collect and visualize NERSC projects
      * Simplify and automate NERSC-10 interoperability with other data center resources
      * Increase ability to support emerging technologies
      * Implement continuous improvement to data center without major disruptions to users
    * **Long-term (6-10 years)**:
      * Achieve pervasive AI, where AI is used to accelerate science and HPC wherever possible
      * Enable scientists to use AI with human and AI-driven expertise
      * Develop applications for science with large-scale, science-informed, robust, transferable models
      * Train staff in using modern hardware, software, and techniques to support AI adoption
      * Mesh operational efforts seamlessly with forward-looking strategies, including NERSC-11
    
    This implementation plan is based on the specific goals and timelines mentioned in the NERSC Strategic Plan FY2024-2032, and is intended to provide a general roadmap for achieving the strategic AI goals outlined in the plan.



```python
answer = qa_chain.invoke(
    "Based on the information within the pdf and NERSC's AI goals what are the most important challenges in achieving these goals"
)

print(answer)
```

    According to the provided PDF content, NERSC's AI goals are centered around building a pervasive AI ecosystem that enables scientists to apply large AI models, accelerate scientific discovery, and integrate AI into all science workflows. However, achieving these goals will require overcoming several challenges. Based on the information provided, some of the most important challenges in achieving NERSC's AI goals are:
    
    1. **Scaling AI capabilities without significantly increasing staff effort**: The PDF content mentions that "Expanding the world-class NERSC user experience to meet these challenges (without significantly increasing the NERSC staff effort) will be challenging." This indicates that NERSC faces a challenge in scaling its AI capabilities while maintaining its current staff levels.
    
    2. **Keeping pace with the fast pace of AI development**: The PDF content states that "The fast pace of AI development motivates the continuing" need for innovation and adaptation in NERSC's AI ecosystem. This implies that NERSC must be able to keep pace with the rapid evolution of AI technologies, which can be a significant challenge.
    
    3. **Balancing limited resources to prioritize focus areas**: The PDF content mentions that NERSC has identified four focus areas to prioritize the use of limited resources, including "Impact on science through partnership with users," "System and data center architecture," "Smart green facility," and "Workforce development." This suggests that NERSC faces a challenge in allocating its limited resources effectively across these focus areas to achieve its AI goals.
    
    4. **Designing and implementing next-generation AI services platforms**: The PDF content mentions that one of NERSC's goals is to "Produce initial design and prototypes of components for a next-generation NERSC AI services platform" within the next 1-2 years. This implies that NERSC faces a challenge in designing and implementing a next-generation AI services platform that meets the evolving needs of its users.
    
    5. **Integrating AI into all science workflows**: The PDF content states that NERSC's goal is to "seamlessly integrate AI into all science workflows" by exploiting innovations in various areas of its strategy. This will require significant efforts to adapt existing workflows, develop new tools and frameworks, and train users to effectively leverage AI capabilities.
    
    6. **Providing cutting-edge HPC systems and standardized frameworks for AI**: The PDF content emphasizes the need for "cutting-edge HPC systems for AI together with standardized frameworks and adaptable tools for use in AI training." This implies that NERSC faces a challenge in providing the necessary infrastructure and tools to support AI applications and workflows.
    
    Overall, achieving NERSC's AI goals will require addressing these challenges and making significant progress in various areas, including scaling AI capabilities, keeping pace with AI development, prioritizing focus areas, designing next-generation AI services platforms, integrating AI into science workflows, and providing cutting-edge HPC systems and frameworks for AI.



```python
# Once finished end all jobs
!scancel $(squeue --me --name=vLLM_RAG --format="%i" -h) $(squeue --me --name=embed_rerank_RAG --format="%i" -h)
```


```python

```
