# ATOM: AdapTive and OptiMized Temporal Knowledge Graph Construction

[![GitHub forks](https://img.shields.io/github/forks/geeekai/atom?style=social)](https://github.com/geeekai/atom/fork)
[![GitHub stars](https://img.shields.io/github/stars/geeekai/atom?style=social)](https://github.com/geeekai/atom)

**ATOM** (AdapTive and OptiMized) is a scalable framework for building and continuously updating **Temporal Knowledge Graphs (TKGs)** from timestamped texts. This repository implements a few-shot approach that splits input data into minimal, self-contained “atomic” facts. From these atomic facts, small atomic KGs are derived and then merged **in parallel** to preserve **semantic** and **temporal** consistency at scale.

---

## Key Features

- **Few-Shot Extraction**: No extensive domain-specific fine-tuning is required.
- **Atomic Fact Splitting**: Processes small, self-contained text segments to minimize the “forgetting effect.”
- **Dual-Time Modeling**: Distinguishes between **inherent timestamps** (when the fact actually occurs) and **observation timestamps** (when the fact is ingested).
- **Parallel Architecture**: Merges Knowledge Graphs (KGs) in parallel without repeated LLM prompts for merging.
- **Stable & Exhaustive**: Ensures consistent outputs across multiple LLM runs while capturing as many facts as possible.
- **Continuous Updates**: Seamlessly accommodates new information while preserving previously ingested data.


## ATOM's Architecture

ATOM’s architecture is designed to dynamically construct Temporal Knowledge Graphs from unstructured, time-stamped text by first decomposing documents into atomic facts—small, self-contained segments that reduce the “forgetting effect” in LLMs, and then extracting subject, relation–object triplets along with both inherent and observation timestamps. These extracted triplets form atomic KGs that are merged in parallel using vector-based matching and threshold criteria to resolve entity and relationship conflicts, while a dual-time modeling approach maintains historical records and manages temporal inconsistencies. This modular and parallel design ensures high scalability, robustness, and continuous updates in dynamic, real-world data environments.

<p align="center">
  <img src="./docs/ATOM-archi.png" width="500px" alt="ATOM Workflow Diagram">
</p>

---

## Example of the ATOM Workflow

<p align="center">
  <img src="./docs/atom_flow_example.png" width="500px" alt="ATOM Workflow Diagram">
</p>

1. **Document Distillation**: Input text is split into atomic facts—short, self-contained chunks—by a lightweight documents distiller.
2. **Triplet Extraction**: Each atomic fact is converted into subject–relation–object triplets, including `t_valid` and `t_invalid` timestamps.
3. **Atomic KG Construction**: For each atomic fact, a miniature “atomic KG” is constructed with embedded entities and relationships.
4. **Parallel Merging**: All atomic KGs are merged in parallel using entity- and relation-matching thresholds to maintain consistency.
5. **TKG Updates**: Newly arrived data is merged into the existing TKG without reprocessing older information, enabling dynamic updates.

For more technical details, check out:
- **`atom/atom.py`**: Core logic for building, merging, and updating the knowledge graphs.
- **`evaluation/`**: Notebooks demonstrating performance benchmarks, stability tests, and scalability studies.

---

## Latency & Scalability

<p align="center">
  <img src="./docs/latency.png" width="500px" alt="Latency Comparison">
</p>

- **Parallel Merging**: ATOM’s parallel merging strategy significantly reduces overall latency, as illustrated in the above figure.
- **Scalability**: Merging is performed without additional LLM prompts, making it feasible to scale to millions of nodes in real-world deployments.

---

## Installation

1. **Clone or Fork** the repository:
   ```bash
   git clone https://github.com/geeekai/atom.git
   cd atom


2. **Install Requirements**

Install all dependencies by running:

```bash
pip install -r requirements.txt
```

3. **(Optional) Set Up a Virtual Environment**
It is recommended to use a virtual environment (e.g., conda, venv) to isolate dependencies.

# Example: Building a Temporal Knowledge Graph (TKG) with ATOM from LLMS History

In this example, we demonstrate how to use ATOM to extract factoids from a dataset, build a dynamic Temporal Knowledge Graph (TKG) across different observation timestamps, and finally visualize the graph using Neo4j.

The process involves:
1. **Loading Data**: Reading an Excel file containing LLMS history with associated observation dates.
2. **Factoid Extraction**: Using the `LangchainOutputParser` to extract factoids from the text.
3. **Graph Construction**: Grouping factoids by observation date and building a knowledge graph that merges atomic KGs from different timestamps.
4. **Visualization**: Rendering the final graph using the GraphIntegrator module connected to a Neo4j database.

Below is the derived example code:

---

```python
import pandas as pd
import asyncio
import ast

# Import LLM and Embeddings models using LangChain wrappers
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from atom.utils import LangchainOutputParser, Factoid
from atom import Atom
from atom.graph_integration import GraphIntegrator

# Set up the OpenAI LLM and embeddings models (replace "##" with your API key)
openai_api_key = "##"
openai_llm_model = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt4o-mini",
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

openai_embeddings_model = OpenAIEmbeddings(
    api_key=openai_api_key,
    model="text-embedding-3-large",
)

# Initialize the Langchain output parser with the OpenAI models
lg = LangchainOutputParser(llm_model=openai_llm_model, embeddings_model=openai_embeddings_model)

# Load the LLMS history dataset (ensure the correct path to your Excel file)
llms_history = pd.read_excel("../datasets/llms_history_and_openai_posts/llms_history.xlsx")

# Define a helper function to convert the dataframe's factoids into a dictionary,
# where keys are observation dates and values are the combined list of factoids for that date.
def to_dictionary(df):
    # Convert factoid strings to lists if necessary
    if isinstance(df['factoids'][0], str):
        df["factoids"] = df["factoids"].apply(lambda x: ast.literal_eval(x))
    grouped_df = df.groupby("observation date")["factoids"].sum().reset_index()
    return {
        str(date): factoids
        for date, factoids in grouped_df.set_index("observation date")["factoids"].to_dict().items()
    }

# Convert the LLMS history dataframe into the required dictionary format
llms_history_dict = to_dictionary(llms_history)

# Initialize the ATOM pipeline with the OpenAI models
atom = Atom(llm_model=openai_llm_model, embeddings_model=openai_embeddings_model)

# Build the knowledge graph across different observation timestamps
kg = await atom.build_graph_from_different_obs_times(
    atomic_facts_with_obs_timestamps=llms_history_dict,
    rel_threshold=0.7,
    ent_threshold=0.8
)

# Visualize the resulting knowledge graph using Neo4j
URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "##"
GraphIntegrator(uri=URI, username=USERNAME, password=PASSWORD).visualize_graph(knowledge_graph=kg)
```
# Contributing

We welcome contributions! To help improve ATOM:
	1.	Fork this repository to your GitHub account.
	2.	Create a feature branch with your enhancements or bug fixes.
	3.	Submit a pull request detailing the changes.

Please report any issues via the Issues tab. Community feedback is invaluable!
