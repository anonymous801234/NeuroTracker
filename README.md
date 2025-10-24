# 🧠 Cyber Cortex — Neural Structure Graph Lab

Cyber Cortex is an advanced NLP-powered system that analyzes how **traits**, **environment**, and **neural patterns** co-modulate the brain's functional architecture. It extracts relationships from neuroscience research documents and builds interactive knowledge graphs to visualize neural connectivity and functional relationships.

The system processes scientific text to identify entities like neural regions (hippocampus, cortex, amygdala), neural patterns (theta oscillations, spike activity), environmental factors (stress, novelty, reward), and psychological traits, then maps their relationships in a dynamic graph database.

---

## 🚀 Features

| Capability                           | Description                                                                 |
| ------------------------------------ | --------------------------------------------------------------------------- |
| **Document Upload**                  | Supports PDF, DOCX, and TXT research documents                              |
| **Intelligent Text Extraction**      | Automatic text extraction with format detection                             |
| **Neuroscience-Aware Preprocessing** | Uses SciSpaCy scientific NLP model for entity recognition and normalization |
| **Relationship Extraction**          | Extracts (subject, relation, object) triples with confidence scores         |
| **Dual Storage Options**             | Choose between Neo4j graph database or local file storage (JSON/CSV)        |
| **Interactive Graph Visualization**  | Real-time visualization of neural relationships and connectivity            |
| **Entity Classification**            | Automatic categorization: NEURAL_REGION, NEURAL_PATTERN, ENVIRONMENT, TRAIT |
| **Connection Testing**               | Built-in Neo4j connection testing with detailed troubleshooting guides      |

## 📋 Prerequisites

- Python 3.8+
- Neo4j Desktop (for graph database) or Neo4j AuraDB (cloud)
- Required Python packages (see Installation)

## 🛠️ Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd neurotracker
   ```

2. **Install Python dependencies:**

   ```bash
   pip install streamlit pandas pdfplumber python-docx neo4j plotly
   ```

3. **Install SciSpaCy scientific NLP model:**

   ```bash
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz
   ```

4. **Set up Neo4j Database:**

   - **Option A: Neo4j Desktop (Local)**

     - Download and install Neo4j Desktop
     - Create a new database
     - Start the database (default URI: `bolt://localhost:7687`)
     - Default credentials: username `neo4j`, password as set during creation

   - **Option B: Neo4j AuraDB (Cloud)**
     - Sign up at [Neo4j AuraDB Console](https://console.neo4j.io)
     - Create a free database
     - Copy connection details (URI, username, password)

## 🚀 Usage

1. **Start the application:**

   ```bash
   streamlit run main1.py
   ```

2. **Upload a document:**

   - Click "Choose a file" and select a PDF, DOCX, or TXT file
   - The system will automatically extract text from the document

3. **Preprocess the text:**

   - Click the "🔧 Preprocess" button to analyze the text
   - The system will identify entities and prepare for relationship extraction

4. **Configure storage:**

   - Choose between "Local File" or "Neo4j Database" storage
   - For Neo4j, enter your connection details and test the connection

5. **Generate the graph:**
   - Click "🕸️ Generate Graph" to extract relationships and build the knowledge graph
   - View the interactive visualization in the right panel

## 📊 How It Works

1. **Text Processing:** Documents are parsed and cleaned using scientific NLP techniques
2. **Entity Recognition:** SciSpaCy identifies neuroscience-related entities
3. **Relationship Extraction:** Custom rules extract relationships between entities
4. **Entity Classification:** Entities are categorized into neural regions, patterns, environments, and traits
5. **Graph Construction:** Relationships are stored in Neo4j or saved as files
6. **Visualization:** Interactive graphs show neural connectivity and functional relationships

## 🔧 Configuration

- **Storage Options:**

  - **Local File:** Saves relationships as JSON or CSV files in the `./output` directory
  - **Neo4j Database:** Stores data in a graph database for complex queries and visualization

- **Entity Types:**
  - **NEURAL_REGION:** Brain areas (hippocampus, cortex, amygdala, PFC)
  - **NEURAL_PATTERN:** Activity patterns (theta, activation, spike, oscillation)
  - **ENVIRONMENT:** External factors (stress, novelty, reward, environment)
  - **TRAIT:** Psychological characteristics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
