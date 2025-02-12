# Multi-Agent LangGraph 3

## Overview
Multi-Agent LangGraph 3 is a project designed to facilitate the integration of multi-agent systems with language models and embedding techniques. This project provides a framework for building applications that leverage embeddings and question-answering capabilities.

## Project Structure
```
repos
└── multi-agent-langgraph-3
    ├── data
    │   ├── links
    │   │   └── example_links.txt
    │   └── index.json
    ├── models
    │   ├── embedding_model
    │   └── llm_model
    ├── src
    │   ├── embedding_model.py
    │   ├── embedding_node.py
    │   ├── qa_node.py
    │   └── main.py
    ├── requirements.txt
    └── README.md
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd multi-agent-langgraph-3
pip install -r requirements.txt
```

## Usage
To run the application, execute the main script:

```bash
python3 src/frontend.py 
```

## Files Description
- **data/links/example_links.txt**: Contains example link data used by the application.
- **data/index.json**: JSON configuration file with structured information relevant to the project.
- **models/embedding_model**: Directory containing the implementation of the embedding model.
- **models/llm_model**: Directory containing the implementation of the language model.
- **src/embedding_model.py**: Implementation of the embedding model, defining classes and functions for embeddings.
- **src/embedding_node.py**: Defines classes/functions related to nodes in the embedding graph.
- **src/qa_node.py**: Manages logic for processing queries and returning answers.
- **src/main.py**: Entry point for the application.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.