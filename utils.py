import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.schema import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_together.chat_models import ChatTogether
from langchain_together.embeddings import TogetherEmbeddings
import spacy
import pandas as pd
import json
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fact_checker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
logger.info("Environment variables loaded")

# ---------- API Key Helper -------------------------------------------------
def get_together_api_key() -> str:
    """Get Together AI API key from environment variables."""
    try:
        key = os.getenv("TOGETHER_API_KEY")
        if key:
            logger.info("Together AI API key found")
            return key

        # If not found, raise error
        error_msg = (
            "TOGETHER_API_KEY not found. Please set it in one of these ways:\n"
            "1. Create a .env file with: TOGETHER_API_KEY=your_key_here\n"
            "2. Set environment variable: export TOGETHER_API_KEY=your_key_here"
        )
        logger.error(error_msg)
        raise EnvironmentError(error_msg)
    except Exception as e:
        logger.exception("Error retrieving Together AI API key")
        raise


# ========================================================================
# FACT-CHECKING SYSTEM COMPONENTS (OOP Architecture)
# ========================================================================

class ClaimExtractor:
    """
    Handles claim and entity extraction using NLP (spaCy).
    Follows Single Responsibility Principle.
    """

    # Supported entity types for extraction
    ENTITY_TYPES = ['ORG', 'GPE', 'PERSON', 'DATE', 'EVENT', 'MONEY',
                    'PERCENT', 'LAW', 'PRODUCT']

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the ClaimExtractor with a spaCy model.

        Args:
            model_name: Name of the spaCy model to use
        """
        self.model_name = model_name
        self._nlp = None

    @property
    def nlp(self):
        """Lazy load spaCy model to avoid startup overhead."""
        if self._nlp is None:
            try:
                logger.info(f"Loading spaCy model: {self.model_name}")
                self._nlp = spacy.load(self.model_name)
                logger.info(f"Successfully loaded spaCy model: {self.model_name}")
            except OSError as e:
                logger.error(f"spaCy model '{self.model_name}' not found")
                raise RuntimeError(
                    f"spaCy model '{self.model_name}' not found. "
                    f"Please install it with: python -m spacy download {self.model_name}"
                )
            except Exception as e:
                logger.exception(f"Unexpected error loading spaCy model: {self.model_name}")
                raise
        return self._nlp

    def extract_entities(self, doc) -> List[Dict[str, Any]]:
        """
        Extract named entities from a spaCy document.

        Args:
            doc: spaCy document object

        Returns:
            List of entity dictionaries with text, type, and position
        """
        try:
            entities = []
            for ent in doc.ents:
                if ent.label_ in self.ENTITY_TYPES:
                    entities.append({
                        'text': ent.text,
                        'type': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
            logger.debug(f"Extracted {len(entities)} entities")
            return entities
        except Exception as e:
            logger.exception("Error extracting entities")
            return []

    def extract_claims(self, text: str, min_length: int = 10) -> List[Dict[str, Any]]:
        """
        Extract key claims and named entities from input text.

        Args:
            text: Input text (e.g., news post, social media statement)
            min_length: Minimum length for a sentence to be considered a claim

        Returns:
            List of claim dictionaries with 'text', 'type', and 'entities'
        """
        try:
            logger.info(f"Extracting claims from text ({len(text)} chars)")
            doc = self.nlp(text)
            entities = self.extract_entities(doc)

            # Extract sentences as potential claims
            claims = []
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if len(sent_text) >= min_length:
                    # Find entities in this sentence
                    sent_entities = [
                        e for e in entities
                        if e['start'] >= sent.start_char and e['end'] <= sent.end_char
                    ]

                    claims.append({
                        'text': sent_text,
                        'type': 'statement',
                        'entities': sent_entities
                    })

            # If no claims extracted, treat entire text as one claim
            if not claims:
                logger.debug("No sentences found, using entire text as claim")
                claims.append({
                    'text': text.strip(),
                    'type': 'statement',
                    'entities': entities
                })

            logger.info(f"Extracted {len(claims)} claim(s)")
            return claims
        except Exception as e:
            logger.exception("Error extracting claims")
            # Return fallback claim
            return [{
                'text': text.strip(),
                'type': 'statement',
                'entities': []
            }]


class FactsDatabase:
    """
    Manages the verified facts database and vector store.
    Handles loading, embedding, and persistence.
    """

    DEFAULT_CSV_PATH = "tests/verified_facts_db.csv"
    DEFAULT_INDEX_PATH = "faiss_index_facts"
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

    def __init__(self, api_key: str = None):
        """
        Initialize the FactsDatabase.

        Args:
            api_key: Together AI API key (optional, can use get_together_api_key)
        """
        logger.info("Initializing FactsDatabase")
        self.api_key = api_key or get_together_api_key()

        try:
            self.embeddings = TogetherEmbeddings(
                model=self.EMBEDDING_MODEL,
                api_key=self.api_key
            )
            logger.info(f"Embeddings initialized with model: {self.EMBEDDING_MODEL}")

            # Initialize ClaimExtractor for entity extraction from facts
            self.claim_extractor = ClaimExtractor()
            logger.info("ClaimExtractor initialized for database entity extraction")

        except Exception as e:
            logger.exception("Error initializing embeddings")
            raise

    def load_from_csv(
        self,
        csv_path: str = None,
        index_path: str = None
    ) -> str:
        """
        Load verified facts from CSV and create FAISS vector store.

        Args:
            csv_path: Path to verified facts CSV file
            index_path: Path to save FAISS index

        Returns:
            Status message with count of loaded facts
        """
        csv_path = csv_path or self.DEFAULT_CSV_PATH
        index_path = index_path or self.DEFAULT_INDEX_PATH

        try:
            logger.info(f"Loading facts from CSV: {csv_path}")
            # Read verified facts
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} rows from CSV")

            # Handle different CSV formats
            if 'fact_text' in df.columns:
                fact_column = 'fact_text'
                logger.debug("Using 'fact_text' column")
            elif 'fact' in df.columns:
                fact_column = 'fact'
                logger.debug("Using 'fact' column")
            else:
                error_msg = "CSV must contain a 'fact' or 'fact_text' column"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Create documents with metadata
            logger.info("Creating documents with metadata")
            documents = self._create_documents(df, fact_column)
            logger.info(f"Created {len(documents)} documents")

            # Create FAISS index
            logger.info("Creating FAISS vector index...")
            vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info("FAISS index created successfully")

            # Save to disk
            logger.info(f"Saving FAISS index to: {index_path}")
            vector_store.save_local(index_path)
            logger.info("FAISS index saved successfully")

            return f"✅ Successfully loaded {len(documents)} verified facts into vector store"

        except FileNotFoundError:
            raise FileNotFoundError(f"Verified facts CSV not found at: {csv_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading verified facts: {str(e)}")

    def _create_documents(
        self,
        df: pd.DataFrame,
        fact_column: str
    ) -> List[LangchainDocument]:
        """
        Create LangChain documents from DataFrame with entity extraction.

        Args:
            df: Pandas DataFrame with facts
            fact_column: Name of the column containing fact text

        Returns:
            List of LangChain documents with metadata including extracted entities
        """
        try:
            documents = []
            multi_sentence_count = 0
            pronoun_count = 0

            for idx, row in df.iterrows():
                fact_text = row[fact_column]

                # Extract fact_id if available
                if 'fact_id' in df.columns:
                    fact_id = row['fact_id']
                else:
                    fact_id = f"F{idx:03d}"

                # DATA VALIDATION: Check for multi-sentence facts
                sentences = fact_text.split('.')
                if len([s for s in sentences if s.strip()]) > 1:
                    multi_sentence_count += 1
                    logger.warning(
                        f"Fact {fact_id} contains multiple sentences ({len(sentences)} sentences). "
                        f"Consider splitting for better retrieval: {fact_text[:80]}..."
                    )

                # DATA VALIDATION: Check for unresolved pronouns
                pronouns = ['he ', 'she ', 'it ', 'they ', 'them ', 'his ', 'her ', 'their ']
                if any(pronoun in fact_text.lower() for pronoun in pronouns):
                    pronoun_count += 1
                    logger.warning(
                        f"Fact {fact_id} contains pronouns - may cause coreference issues: {fact_text[:80]}..."
                    )

                # ENTITY EXTRACTION: Extract entities from fact text
                entities = []
                entities_dict = {}
                try:
                    claims = self.claim_extractor.extract_claims(fact_text)
                    if claims and len(claims) > 0:
                        entities = claims[0].get('entities', [])
                        # Convert entities list to dict for easier access
                        entities_dict = {
                            'organizations': [e['text'] for e in entities if e['type'] in ['ORG', 'ORGANIZATION']],
                            'locations': [e['text'] for e in entities if e['type'] in ['GPE', 'LOC', 'LOCATION']],
                            'persons': [e['text'] for e in entities if e['type'] in ['PERSON', 'PER']],
                            'dates': [e['text'] for e in entities if e['type'] == 'DATE'],
                            'percentages': [e['text'] for e in entities if e['type'] in ['PERCENT', 'PERCENTAGE']],
                            'money': [e['text'] for e in entities if e['type'] in ['MONEY', 'CURRENCY']],
                            'all_entities': [e['text'] for e in entities]
                        }
                        logger.debug(f"Fact {fact_id}: Extracted {len(entities)} entities")
                except Exception as e:
                    logger.warning(f"Failed to extract entities from fact {fact_id}: {str(e)}")

                # Create metadata with entities
                metadata = {
                    'source': row.get('source', 'Verified Database'),
                    'date': row.get('date', 'N/A'),
                    'category': row.get('category', 'General'),
                    'fact_id': fact_id,
                    'entities': entities,  # Full entity list with types
                    'entities_dict': entities_dict  # Organized by type for easy filtering
                }

                # Create LangChain document with metadata
                doc = LangchainDocument(
                    page_content=fact_text,
                    metadata=metadata
                )
                documents.append(doc)

            # Summary logging
            logger.info(f"Created {len(documents)} documents from DataFrame")
            if multi_sentence_count > 0:
                logger.warning(
                    f"⚠️  {multi_sentence_count}/{len(documents)} facts contain multiple sentences. "
                    f"Consider atomic splitting for better granularity."
                )
            if pronoun_count > 0:
                logger.warning(
                    f"⚠️  {pronoun_count}/{len(documents)} facts contain pronouns. "
                    f"Consider coreference resolution."
                )

            # Log entity extraction statistics
            total_entities = sum(len(doc.metadata.get('entities', [])) for doc in documents)
            avg_entities = total_entities / len(documents) if documents else 0
            logger.info(
                f"Entity extraction complete: {total_entities} total entities "
                f"({avg_entities:.1f} avg per fact)"
            )

            return documents
        except Exception as e:
            logger.exception("Error creating documents from DataFrame")
            raise


class FactRetriever:
    """
    Retrieves similar facts from the vector store using semantic search.
    Implements retrieval strategies and similarity scoring.
    """

    DEFAULT_INDEX_PATH = "faiss_index_facts"
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

    def __init__(self, api_key: str = None, index_path: str = None):
        """
        Initialize the FactRetriever.

        Args:
            api_key: Together AI API key
            index_path: Path to FAISS index
        """
        self.api_key = api_key or get_together_api_key()
        self.index_path = index_path or self.DEFAULT_INDEX_PATH
        logger.info(f"Initializing FactRetriever with index path: {self.index_path}")

        try:
            self.embeddings = TogetherEmbeddings(
                model=self.EMBEDDING_MODEL,
                api_key=self.api_key
            )
            logger.info(f"Embeddings model initialized: {self.EMBEDDING_MODEL}")
        except Exception as e:
            logger.exception("Error initializing embeddings model")
            raise

        self._vector_store = None

    @property
    def vector_store(self):
        """Lazy load vector store to avoid unnecessary I/O."""
        if self._vector_store is None:
            try:
                logger.info(f"Loading FAISS index from: {self.index_path}")
                self._vector_store = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS index loaded successfully")
            except FileNotFoundError:
                error_msg = f"FAISS index not found at: {self.index_path}. Please initialize the database first."
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            except Exception as e:
                logger.exception("Error loading FAISS index")
                raise RuntimeError(f"Error loading FAISS index: {str(e)}")
        return self._vector_store

    def retrieve(
        self,
        claim: str,
        top_k: int = 3,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most similar verified facts for a given claim.

        Args:
            claim: The claim text to verify
            top_k: Number of similar facts to retrieve
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of dictionaries with 'fact', 'metadata', and 'similarity'
        """
        try:
            logger.info(f"Retrieving top-{top_k} facts for claim: {claim[:100]}...")

            # Perform similarity search with scores
            docs_with_scores = self.vector_store.similarity_search_with_score(
                claim, k=top_k
            )
            logger.debug(f"Retrieved {len(docs_with_scores)} documents from FAISS")

            # Format and filter results
            similar_facts = []
            for doc, score in docs_with_scores:
                # FAISS returns distance, convert to similarity
                similarity = self._normalize_similarity(score)

                if similarity >= similarity_threshold:
                    similar_facts.append({
                        'fact': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity': round(similarity, 3)
                    })
                    logger.debug(f"Fact similarity: {similarity:.3f} - {doc.page_content[:50]}...")

            logger.info(f"Filtered to {len(similar_facts)} facts above threshold {similarity_threshold}")
            return similar_facts

        except Exception as e:
            logger.exception("Error retrieving similar facts")
            raise RuntimeError(f"Error retrieving similar facts: {str(e)}")

    @staticmethod
    def _normalize_similarity(distance: float) -> float:
        """
        Convert FAISS distance to similarity score (0-1 range).

        Args:
            distance: FAISS distance score (lower = more similar)

        Returns:
            Normalized similarity score
        """
        return 1 / (1 + distance)


class ClaimClassifier:
    """
    Uses LLM to classify claims as True/False/Unverifiable.
    Handles prompt engineering and response parsing.
    """

    LLM_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    TEMPERATURE = 0.3

    # Verdict constants
    VERDICT_TRUE = "Likely True"
    VERDICT_FALSE = "Likely False"
    VERDICT_UNVERIFIABLE = "Unverifiable"

    def __init__(self, api_key: str = None):
        """
        Initialize the ClaimClassifier.

        Args:
            api_key: Together AI API key
        """
        self.api_key = api_key or get_together_api_key()
        logger.info(f"Initializing ClaimClassifier with model: {self.LLM_MODEL}")

        try:
            self.llm = ChatTogether(
                model=self.LLM_MODEL,
                temperature=self.TEMPERATURE,
                api_key=self.api_key
            )
            logger.info(f"LLM initialized successfully (temperature={self.TEMPERATURE})")
        except Exception as e:
            logger.exception("Error initializing LLM")
            raise

    def classify(
        self,
        claim: str,
        retrieved_facts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Classify a claim against retrieved facts using LLM.

        Args:
            claim: The original claim to verify
            retrieved_facts: List of similar facts with metadata

        Returns:
            Dictionary with 'verdict', 'confidence', 'reasoning', 'evidence_used'
        """
        logger.info(f"Classifying claim with {len(retrieved_facts)} retrieved facts")

        # Build prompt with evidence
        prompt = self._build_prompt(claim, retrieved_facts)
        logger.debug(f"Built prompt with {len(prompt)} characters")

        try:
            # Get LLM response
            logger.info("Invoking LLM for claim classification")
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            response_text = response.content.strip()
            logger.debug(f"LLM response received ({len(response_text)} chars)")

            # Parse JSON response
            result = self._parse_response(response_text)
            logger.info(f"Classification result: {result['verdict']} (confidence: {result['confidence']})")

            # Add retrieved facts as evidence details
            result['evidence_details'] = retrieved_facts

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            return self._fallback_response(retrieved_facts, "JSON parsing failed")
        except Exception as e:
            logger.exception("Error during claim classification")
            return self._fallback_response(retrieved_facts, str(e))

    def _build_prompt(
        self,
        claim: str,
        retrieved_facts: List[Dict[str, Any]]
    ) -> str:
        """
        Build the classification prompt for the LLM.

        Args:
            claim: The claim to verify
            retrieved_facts: Retrieved evidence

        Returns:
            Formatted prompt string
        """
        # Format evidence
        evidence_text = self._format_evidence(retrieved_facts)

        # Construct prompt
        prompt = f"""You are a fact-checking assistant. Your task is to verify the following claim against verified evidence.

CLAIM TO VERIFY:
"{claim}"

VERIFIED EVIDENCE FROM DATABASE:
{evidence_text}

INSTRUCTIONS:
1. Compare the claim against the verified evidence carefully
2. Classify the claim as one of:
   - "{self.VERDICT_TRUE}" - if evidence strongly supports the claim
   - "{self.VERDICT_FALSE}" - if evidence contradicts the claim
   - "{self.VERDICT_UNVERIFIABLE}" - if insufficient or conflicting evidence

3. Provide your analysis in EXACTLY this JSON format (no additional text):
{{
  "verdict": "{self.VERDICT_TRUE}" | "{self.VERDICT_FALSE}" | "{self.VERDICT_UNVERIFIABLE}",
  "confidence": "high" | "medium" | "low",
  "reasoning": "Explain your decision in 2-3 sentences",
  "evidence_used": ["fact 1", "fact 2"]
}}

IMPORTANT:
- Be objective and base your verdict only on the evidence provided
- If the evidence is vague or irrelevant, mark as "{self.VERDICT_UNVERIFIABLE}"
- Consider dates, entities, and specific details when comparing
- Return ONLY the JSON object, no other text

YOUR RESPONSE:"""

        return prompt

    def _format_evidence(self, retrieved_facts: List[Dict[str, Any]]) -> str:
        """
        Format retrieved facts for the prompt.

        Args:
            retrieved_facts: List of facts with metadata

        Returns:
            Formatted evidence string
        """
        if not retrieved_facts:
            return "No similar verified facts found in the database."

        evidence_lines = []
        for i, fact in enumerate(retrieved_facts, 1):
            lines = [
                f"Evidence {i}:",
                f"{fact['fact']}",
                f"Source: {fact['metadata'].get('source', 'Unknown')}",
                f"Date: {fact['metadata'].get('date', 'Unknown')}",
                f"Similarity: {fact['similarity']:.2f}"
            ]
            evidence_lines.append("\n".join(lines))

        return "\n\n".join(evidence_lines)

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM JSON response.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed result dictionary
        """
        try:
            # Try to extract JSON if LLM added extra text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
                logger.debug("Extracted JSON from LLM response")

            result = json.loads(response_text)
            logger.debug("Successfully parsed JSON response")

            # Validate required fields
            required_fields = ['verdict', 'confidence', 'reasoning', 'evidence_used']
            missing_fields = [field for field in required_fields if field not in result]

            if missing_fields:
                logger.warning(f"Missing fields in LLM response: {missing_fields}")
                for field in missing_fields:
                    result[field] = "Unknown" if field != 'evidence_used' else []

            return result
        except Exception as e:
            logger.exception("Error parsing LLM response")
            raise

    def _fallback_response(
        self,
        retrieved_facts: List[Dict[str, Any]],
        error_msg: str
    ) -> Dict[str, Any]:
        """
        Create fallback response on error.

        Args:
            retrieved_facts: Retrieved evidence
            error_msg: Error message

        Returns:
            Fallback response dictionary
        """
        logger.warning(f"Creating fallback response due to: {error_msg}")
        return {
            'verdict': self.VERDICT_UNVERIFIABLE,
            'confidence': 'low',
            'reasoning': f'Error during fact-checking: {error_msg}',
            'evidence_used': [],
            'evidence_details': retrieved_facts,
            'error': error_msg
        }


class FactChecker:
    """
    Main orchestrator for the fact-checking pipeline.
    Coordinates ClaimExtractor, FactRetriever, and ClaimClassifier.
    Follows Facade pattern to provide simple interface.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the FactChecker with all required components.

        Args:
            api_key: Together AI API key
        """
        logger.info("Initializing FactChecker pipeline")
        self.api_key = api_key or get_together_api_key()

        try:
            # Initialize components (Dependency Injection)
            logger.debug("Initializing ClaimExtractor")
            self.claim_extractor = ClaimExtractor()

            logger.debug("Initializing FactRetriever")
            self.fact_retriever = FactRetriever(api_key=self.api_key)

            logger.debug("Initializing ClaimClassifier")
            self.claim_classifier = ClaimClassifier(api_key=self.api_key)

            logger.info("FactChecker initialization complete")
        except Exception as e:
            logger.exception("Error initializing FactChecker")
            raise

    def check_claim(self, user_claim: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Main fact-checking pipeline that orchestrates the entire process.

        Args:
            user_claim: User's input claim/statement to verify
            top_k: Number of similar facts to retrieve

        Returns:
            Complete fact-check result with verdict, evidence, and reasoning
        """
        logger.info("=" * 60)
        logger.info(f"Starting fact-check pipeline for claim: {user_claim[:100]}...")
        logger.info("=" * 60)

        try:
            # Step 1: Extract claims from input
            logger.info("Step 1: Extracting claims from input")
            claims = self.claim_extractor.extract_claims(user_claim)

            # For simplicity, fact-check the first/main claim
            main_claim = claims[0]['text'] if claims else user_claim
            logger.info(f"Main claim identified: {main_claim[:100]}...")

            # Step 2: Retrieve similar facts
            logger.info(f"Step 2: Retrieving top-{top_k} similar facts")
            similar_facts = self.fact_retriever.retrieve(main_claim, top_k=top_k)
            logger.info(f"Retrieved {len(similar_facts)} similar facts")

            # Step 3: Classify using LLM
            logger.info("Step 3: Classifying claim using LLM")
            result = self.claim_classifier.classify(main_claim, similar_facts)

            # Step 4: Add metadata
            logger.info("Step 4: Adding metadata to result")
            result['original_input'] = user_claim
            result['extracted_claim'] = main_claim
            result['entities_found'] = claims[0].get('entities', []) if claims else []
            result['total_claims_extracted'] = len(claims)

            logger.info(f"Fact-check complete: {result['verdict']}")
            logger.info("=" * 60)
            return result

        except Exception as e:
            logger.exception("Error in fact-checking pipeline")
            logger.info("=" * 60)
            return self._error_response(user_claim, str(e))

    def _error_response(self, user_claim: str, error_msg: str) -> Dict[str, Any]:
        """
        Create error response when pipeline fails.

        Args:
            user_claim: Original user claim
            error_msg: Error message

        Returns:
            Error response dictionary
        """
        logger.error(f"Creating error response for claim: {error_msg}")
        return {
            'verdict': 'Unverifiable',
            'confidence': 'low',
            'reasoning': f'Error during fact-checking pipeline: {error_msg}',
            'evidence_used': [],
            'evidence_details': [],
            'original_input': user_claim,
            'extracted_claim': user_claim,
            'entities_found': [],
            'error': error_msg
        }


# ========================================================================
# LEGACY FUNCTION WRAPPERS (for backward compatibility)
# ========================================================================

def load_verified_facts(csv_path: str = "tests/verified_facts_db.csv") -> str:
    """
    Legacy wrapper for backward compatibility.
    Uses FactsDatabase class internally.

    Args:
        csv_path: Path to verified facts CSV file

    Returns:
        Status message
    """
    db = FactsDatabase()
    return db.load_from_csv(csv_path)


def retrieve_similar_facts(
    claim: str,
    top_k: int = 3,
    similarity_threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Legacy wrapper for backward compatibility.
    Uses FactRetriever class internally.

    Args:
        claim: The claim text to verify
        top_k: Number of similar facts to retrieve
        similarity_threshold: Minimum similarity score (0-1)

    Returns:
        List of dictionaries with 'fact', 'metadata', and 'similarity'
    """
    retriever = FactRetriever()
    return retriever.retrieve(claim, top_k, similarity_threshold)


def classify_claim(claim: str, retrieved_facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Legacy wrapper for backward compatibility.
    Uses ClaimClassifier class internally.

    Args:
        claim: The original claim to verify
        retrieved_facts: List of similar facts with metadata

    Returns:
        Dictionary with 'verdict', 'confidence', 'reasoning', 'evidence_used'
    """
    classifier = ClaimClassifier()
    return classifier.classify(claim, retrieved_facts)


def fact_check_claim(user_claim: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Legacy wrapper for backward compatibility.
    Uses FactChecker class internally.

    Args:
        user_claim: User's input claim/statement to verify
        top_k: Number of similar facts to retrieve

    Returns:
        Complete fact-check result with verdict, evidence, and reasoning
    """
    checker = FactChecker()
    return checker.check_claim(user_claim, top_k)