# Fact-Checker RAG System

*AI-Powered Claim Verification Against Trusted Sources*

**Fact-Checker** is an intelligent Retrieval-Augmented Generation (RAG) system designed to verify claims from news posts and social media statements against a database of verified facts. The system extracts key claims using NLP, retrieves similar verified facts using semantic search, and uses an LLM to classify claims as True, False, or Unverifiable with supporting evidence and reasoning.

---

## Objective

Build a lightweight system that can analyze a short news post or social media statement, extract key claims or entities, and verify them against a vector database of verified facts using a RAG pipeline. The system classifies claims as:

- ✅ **Likely True** - Evidence supports the claim
- ❌ **Likely False** - Evidence contradicts the claim
- 🤷‍♂️ **Unverifiable** - Insufficient or conflicting evidence

---

## Example

**Input Claim:**
> "The Indian government has announced free electricity to all farmers starting July 2025."

**System Output:**
```json
{
  "verdict": "Likely False",
  "confidence": "medium",
  "reasoning": "The retrieved evidence shows no official announcement about free electricity to farmers starting July 2025. The most similar facts relate to different agricultural policies but do not mention this specific claim.",
  "evidence": [
    "The government announced a subsidy scheme for agricultural equipment in June 2024.",
    "Electricity tariff reforms for farmers were discussed in the 2023 budget."
  ]
}
```

---

## Live Demo

Webapp hosted on Hugging Face Spaces: [*Live Demo*](https://huggingface.co/spaces/adi-123/Fact_Checker)

Explainer video: https://youtu.be/JCi2f47yOag?si=4bb5uSTaIEodvR5l

### 1. Initialize Database
- Navigate to the **Configuration** tab
- Click **"Initialize Fact Database"**
- This loads verified facts from `tests/verified_facts_db.csv` into a FAISS vector store
- The database contains 50 verified statements from trusted sources (e.g., PIB India)
- During initialization, spaCy extracts entities from each database fact for enhanced retrieval
- System performs automatic data quality validation (multi-sentence facts, pronoun detection)

### 2. Enter a Claim
- Go to the **Fact Check** tab
- Type or paste a claim in the text area
- OR select a sample from the dropdown menu (loaded from `tests/social_media_feed.csv`)

### 3. Verify the Claim
- Click **"Check Fact"** to analyze the claim
- The system will:
  1. Extract key claims and entities from user input using spaCy NLP
  2. Retrieve top-3 similar verified facts from the vector database
  3. Compare claim entities with database fact entities
  4. Use LLM (Llama 3.1) to classify the claim with entity-level reasoning
  5. Display verdict with evidence and reasoning

### 4. Review Results
- **Verdict**: Likely True, Likely False, or Unverifiable
- **Confidence**: High/Medium/Low indicator based on evidence quality
- **Entities Extracted**: Key entities detected from claim (organizations, locations, dates, percentages, etc.)
- **Reasoning**: AI-generated explanation for the verdict
- **Evidence**: Retrieved facts with source, date, similarity scores, and extracted entities
- **Entity Matching**: See which entities match between claim and evidence

---

## Architecture
![Architecture](https://github.com/Aditya-NeuralNetNinja/FactChecker/blob/main/architecture.svg)



### System Pipeline

```
User Input (Claim)
    ↓
1. Claim/Entity Extraction (spaCy)
    ↓
2. Semantic Search (FAISS Vector Store)
    ↓
3. Retrieve Top-K Similar Facts
    ↓
4. LLM-Powered Classification (Llama 3.1)
    ↓
5. Output: Verdict + Evidence + Reasoning
```

### Key Components

#### 1. **Claim/Entity Extraction**
- Uses spaCy `en_core_web_sm` NLP model
- Extracts named entities: Organizations, Locations, Persons, Dates, Percentages, Money, Events
- Identifies key claims in multi-sentence inputs
- **NEW**: Symmetric processing - entities extracted from BOTH user claims AND database facts
- Enables entity-level comparison and richer LLM context

#### 2. **Verified Facts Database**
- CSV file with verified facts from trusted sources (PIB India, government press releases)
- Columns: `fact`, `source`, `date`, `category`
- **Enhanced**: Each fact enriched with extracted entities during database initialization
- 114 entities extracted from 50 facts (avg 2.3 entities per fact)
- Embedded using BAAI/bge-base-en-v1.5 model via Together AI
- **Automatic validation**: Detects multi-sentence facts (42%) and pronoun issues (58%)

#### 3. **Vector Store & Retrieval**
- FAISS (Facebook AI Similarity Search) for efficient semantic search
- Embeddings: BAAI/bge-base-en-v1.5 (768-dimensional vectors)
- Retrieves top-k most similar facts with similarity scores

#### 4. **LLM-Powered Classification**
- Model: Meta-Llama-3.1-8B-Instruct-Turbo (via Together AI)
- Temperature: 0.3 (deterministic, focused responses)
- Structured JSON output with verdict, confidence, reasoning, evidence
- Fallback error handling for robustness

#### 5. **Gradio UI**
- Clean, tab-based interface (Fact Check, Configuration, About)
- Sample claims dropdown for quick testing
- Color-coded verdicts and confidence indicators
- Markdown-formatted results with metadata
- Built-in public sharing capability

### Models Used

| Component | Model/Tool | Provider | Purpose |
|-----------|------------|----------|---------|
| **Embedding** | BAAI/bge-base-en-v1.5 | Together AI | Text-to-vector conversion |
| **LLM** | Meta-Llama-3.1-8B-Instruct-Turbo | Together AI | Claim classification |
| **Vector DB** | FAISS | Facebook AI | Semantic search |
| **NLP** | en_core_web_sm | spaCy | Entity extraction |

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Together AI API key (get one at [together.ai](https://www.together.ai/))

### Setup Steps

**1. Clone and navigate to the project:**
```bash
git clone <repository-url>
cd FactChecker
```

**2. Create a virtual environment (recommended):**
```bash
python3 -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Configure API key:**

Create a `.env` file in the project root:
```bash
TOGETHER_API_KEY=your_together_api_key_here
```

**5. Verify test data exists:**

Ensure these files are present:
- `tests/verified_facts_db.csv` - Verified facts (30-50 facts)
- `tests/social_media_feed.csv` - Sample claims for testing

**6. Run the application:**
```bash
python app.py
```

**7. Open in browser:**

The app will automatically open at `http://localhost:7860`

**8. Initialize the database:**
- Go to the **Configuration** tab
- Click **"Initialize Fact Database"**
- Wait for success message
- Start fact-checking in the **Fact Check** tab!

---

## Project Structure

```
FactChecker/
├── app.py                          # Gradio UI application
├── utils.py                        # Core RAG pipeline (5 OOP classes)
├── requirements.txt                # Python dependencies
├── tests/
│   ├── verified_facts_db.csv      # Verified facts database (50 facts)
│   └── social_media_feed.csv      # Sample claims for testing
├── README.md             
```

---

## Data Sources

### Verified Facts Database
- **Source**: Government press releases (e.g., [PIB India RSS](https://www.pib.gov.in/ViewRss.aspx))
- **Format**: CSV with verified statements, enriched with extracted entities
- **Size**: 50 facts 
- **Entity Types**: Organizations, Locations, Dates, Percentages, Money, Persons
- **Example Row**:
  ```
  fact,source,date,category
  "India's GDP growth rate was 7.2% in Q2 2024","PIB India","2024-08-15","Economy"
  ```

- **Example Metadata** (auto-generated during initialization):

  ```python
  {
    'entities_dict': {
      'locations': ['India'],
      'percentages': ['7.2%'],
      'dates': ['Q2 2024']
    }
  }
  ```

### Social Media Test Feed
- **Purpose**: Testing and demonstration
- **Format**: CSV with mix of true/false/ambiguous claims
- **Used for**: Sample claims in UI and system evaluation

---

## Technical Highlights

### NLP & Embedding-Level Thinking

- spaCy for efficient entity extraction with minimal latency (5-10ms per fact)
- **Symmetric entity processing**: entities extracted from BOTH claims AND database facts
- Entity-level metadata enables richer LLM reasoning and future hybrid search
- BGE embeddings for high-quality semantic similarity
- FAISS for fast vector search (sub-100ms on 50 facts)
- L2 distance with normalization (not cosine similarity) for better numerical accuracy

### LLM System Engineering
- Structured prompts with explicit classification instructions
- JSON output parsing with fallback error handling
- Temperature tuning (0.3) for consistent, deterministic responses
- Graceful degradation on API failures

### Logging & Monitoring

- Comprehensive logging system with file and console output
- Logs written to `fact_checker.log` for debugging and audit trails
- Structured logging with timestamps, module names, and severity levels
- Detailed pipeline tracking: initialization, retrieval, classification, and errors
- Exception logging with full stack traces for troubleshooting

### Real-World Practicality
- Works offline after initial fact database setup (FAISS is local)
- Handles edge cases: no matches, malformed input, API failures
- User-friendly error messages and feedback
- Fast response times (<5s for full pipeline)

---

## Features

✅ **NLP & Embeddings**

- **Symmetric entity extraction**: spaCy processes both user claims AND database facts
- Entity types: ORG, GPE, PERSON, DATE, PERCENT, MONEY, EVENT
- Entity-level metadata enriches LLM context for better reasoning
- Automatic data quality validation (multi-sentence facts, pronoun detection)
- Semantic similarity with BGE embeddings (768-dim vectors)
- FAISS vector search with L2 distance normalization (sub-100ms retrieval)

✅ **LLM Engineering**

- Structured prompts with explicit classification format
- JSON parsing with fallback error handling
- Temperature tuning (0.3) for deterministic responses

✅ **Production-Ready Code**

- OOP architecture with 5 classes following SOLID principles
- Type hints and comprehensive docstrings
- Separation of concerns (UI vs business logic)
- Error handling at every layer
- Comprehensive logging system for debugging and monitoring
- **NEW**: Symmetric processing architecture - no asymmetry between query and database

✅ **User Experience**

- Clean Gradio interface with 3 tabs
- Sample claims for quick testing
- Color-coded verdicts and confidence indicators
- Detailed evidence with source attribution and extracted entities
- Entity matching display for transparency
- Auto-initialization on startup (HuggingFace Spaces ready)

---

## Example Output

### Input
```
"India's inflation rate dropped to 4.5% in September 2024"
```

### Output
```json
{
  "verdict": "Likely True",
  "confidence": "high",
  "reasoning": "The retrieved evidence confirms that India's inflation rate was reported at 4.5% in September 2024 according to official government statistics. The claim aligns closely with verified facts from trusted sources.",
  "evidence_used": [
    "India's inflation rate was 4.5% in September 2024 (Source: PIB India, Date: 2024-10-01)"
  ],
  "entities_found": [
    {"text": "India", "type": "GPE"},
    {"text": "4.5%", "type": "PERCENT"},
    {"text": "September 2024", "type": "DATE"}
  ]
}
```

---

## Testing

### Manual Testing
1. Use sample claims from `tests/social_media_feed.csv`
2. Test edge cases:
   - Very vague claims
   - Claims with no matching evidence
   - Multi-sentence complex statements
   - Claims with specific dates/numbers

### Expected Behavior
- **High similarity match** → High confidence verdict
- **No match found** → Unverifiable with low confidence
- **Contradicting evidence** → Likely False verdict
- **Supporting evidence** → Likely True verdict

---

## Limitations

1. **Database Size**: Currently 50 verified facts (proof-of-concept scope, easily expandable)
2. **API Dependency**: Requires Together AI API for embeddings and LLM (can be replaced with local models)
3. **Language**: English only (spaCy model and training data)
4. **Fact Currency**: Facts become outdated; requires periodic refresh
5. **Context Understanding**: May struggle with complex, multi-hop reasoning
