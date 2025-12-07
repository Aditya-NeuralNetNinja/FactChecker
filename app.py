import gradio as gr
import os
import logging
import pandas as pd
from typing import Dict, Any, Tuple, Optional

# Import fact-checking utilities
from utils import (
    load_verified_facts,
    fact_check_claim,
)

# Configure logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------#
#  Gradio Fact-Checking Application
# ---------------------------------------------------------------------------#

# Global state for database initialization
DATABASE_INITIALIZED = False


def initialize_database() -> str:
    """Initialize the verified facts database."""
    global DATABASE_INITIALIZED

    logger.info("Database initialization requested")

    if os.path.exists("faiss_index_facts"):
        DATABASE_INITIALIZED = True
        logger.info("Database already exists at faiss_index_facts/")
        return """
> ## ✅ Database Ready
>
> The fact database is **already initialized** and loaded successfully.
>
> 📁 **Location:** `faiss_index_facts/`

<br>

**🎯 Next Step:** Go to the **Fact Check** tab to start verifying claims!
"""

    try:
        logger.info("Starting database initialization process")
        status_msg = load_verified_facts()
        DATABASE_INITIALIZED = True
        logger.info("Database initialized successfully")
        return f"""
> ## ✅ Initialization Complete
>
> {status_msg}

<br>

**Status:** 🟢 Ready to verify claims

**🎯 Next Step:** Switch to the **Fact Check** tab to start using the system!
"""
    except Exception as e:
        logger.exception("Database initialization failed")
        return f"""
> ## ❌ Initialization Failed
>
> **Error Message:**
> ```
> {str(e)}
> ```

<br>

**🔧 Troubleshooting Steps:**

1. ✓ Ensure `tests/verified_facts_db.csv` exists in your project directory
2. ✓ Verify your `TOGETHER_API_KEY` is set in the `.env` file
3. ✓ Check your internet connection (required for embedding API)
4. ✓ Make sure all dependencies are installed: `pip install -r requirements.txt`

<br>

💡 **Need Help?** Check the README.md for detailed setup instructions.
"""


def load_sample_claims() -> list:
    """Load sample claims from CSV file."""
    logger.info("Loading sample claims")
    sample_claims = []

    if os.path.exists("tests/social_media_feed.csv"):
        try:
            logger.debug("Reading sample claims from CSV")
            df = pd.read_csv("tests/social_media_feed.csv")
            # Handle different column names
            if 'claim' in df.columns:
                sample_claims = df['claim'].tolist()[:10]  # First 10 claims
                logger.info(f"Loaded {len(sample_claims)} sample claims from 'claim' column")
            elif 'text' in df.columns:
                sample_claims = df['text'].tolist()[:10]
                logger.info(f"Loaded {len(sample_claims)} sample claims from 'text' column")
        except Exception as e:
            logger.warning(f"Failed to load sample claims from CSV: {str(e)}")

    # Default samples if CSV not available
    if not sample_claims:
        logger.info("Using default sample claims")
        sample_claims = [
            "The Indian government has announced free electricity to all farmers starting July 2025.",
            "India's GDP growth rate reached 8.2% in Q1 2024.",
            "The Reserve Bank of India has reduced interest rates by 2% this month.",
        ]

    return sample_claims


def format_result(result: Dict[str, Any]) -> str:
    """Format fact-checking results for display."""

    # Verdict with emoji
    verdict = result.get('verdict', 'Unknown')
    verdict_emoji = {
        'Likely True': '✅',
        'Likely False': '❌',
        'Unverifiable': '🤷‍♂️',
        'Unknown': '❓'
    }.get(verdict, '❓')

    # Confidence indicator
    confidence = result.get('confidence', 'unknown')
    confidence_emoji = {
        'high': '🟢',
        'medium': '🟡',
        'low': '🔴'
    }.get(confidence, '⚪')

    # Build output markdown
    output = "# 📊 Fact-Check Results\n\n"
    output += f"> ## {verdict_emoji} Verdict: **{verdict}**\n"
    output += f">\n"
    output += f"> **Confidence Level:** {confidence_emoji} {confidence.capitalize()}\n\n"

    # Extracted claim (if different from input)
    extracted_claim = result.get('extracted_claim', '')
    original_input = result.get('original_input', '')

    if extracted_claim and extracted_claim != original_input:
        output += f"### 🎯 Extracted Claim\n"
        output += f"> {extracted_claim}\n\n"

    # Entities found
    entities = result.get('entities_found', [])
    if entities:
        output += f"### 🏷️ Key Entities Detected\n"
        entity_list = [f"**{e['text']}** ({e['type']})" for e in entities[:5]]
        output += ", ".join(entity_list) + "\n\n"

    # Reasoning
    output += f"### 💭 Reasoning\n"
    reasoning = result.get('reasoning', 'No reasoning provided')
    output += f"> {reasoning}\n\n"

    # Evidence from verified sources
    output += "### 📚 Evidence from Verified Sources\n\n"
    evidence_details = result.get('evidence_details', [])

    if evidence_details:
        for i, evidence in enumerate(evidence_details, 1):
            similarity_pct = evidence['similarity'] * 100
            output += f"#### Evidence #{i}\n"
            output += f"**Similarity Score:** {similarity_pct:.1f}%\n\n"
            output += f"> {evidence['fact']}\n\n"
            metadata = evidence.get('metadata', {})
            output += f"- 📰 **Source:** {metadata.get('source', 'Unknown')}\n"
            output += f"- 📅 **Date:** {metadata.get('date', 'Unknown')}\n"
            output += f"- 🏷️ **Category:** {metadata.get('category', 'General')}\n"

            # Display extracted entities from database fact
            entities_dict = metadata.get('entities_dict', {})
            if entities_dict and any(entities_dict.values()):
                output += f"- 🏷️ **Entities Found:** "
                entity_parts = []
                if entities_dict.get('organizations'):
                    entity_parts.append(f"Orgs: {', '.join(entities_dict['organizations'][:3])}")
                if entities_dict.get('locations'):
                    entity_parts.append(f"Locations: {', '.join(entities_dict['locations'][:3])}")
                if entities_dict.get('dates'):
                    entity_parts.append(f"Dates: {', '.join(entities_dict['dates'][:3])}")
                if entities_dict.get('percentages'):
                    entity_parts.append(f"Percentages: {', '.join(entities_dict['percentages'][:2])}")

                if entity_parts:
                    output += " | ".join(entity_parts)
                else:
                    output += "None"
                output += "\n"

            output += "\n"
    else:
        output += "> ⚠️ No relevant evidence found in the database\n\n"

    # Evidence used by LLM
    evidence_used = result.get('evidence_used', [])
    if evidence_used and evidence_used != ['']:
        output += "<br>\n\n"
        output += "### 🎯 Evidence Cited by AI\n\n"
        for i, ev in enumerate(evidence_used, 1):
            output += f"{i}. {ev}\n"
        output += "\n"

    # Error information (if any)
    if 'error' in result:
        output += f"\n⚠️ **Note:** {result['error']}\n"

    return output


def check_fact(claim: str) -> str:
    """Check a fact claim and return formatted results."""
    global DATABASE_INITIALIZED

    logger.info(f"Fact-check requested for claim: {claim[:100]}...")

    # Check if database is initialized
    if not DATABASE_INITIALIZED:
        if os.path.exists("faiss_index_facts"):
            DATABASE_INITIALIZED = True
            logger.info("Database auto-detected and marked as initialized")
        else:
            logger.warning("Database not initialized - prompting user")
            return "❌ **Error:** Please initialize the fact database first using the button in the Configuration tab."

    # Check if claim is provided
    if not claim or not claim.strip():
        logger.warning("Empty claim provided")
        return "⚠️ **Warning:** Please enter a claim to verify."

    try:
        logger.info("Running fact-checking pipeline")
        # Run fact-checking pipeline
        result = fact_check_claim(claim)

        logger.info(f"Fact-check completed with verdict: {result.get('verdict', 'Unknown')}")
        # Format and return results
        return format_result(result)

    except Exception as e:
        logger.exception("Error during fact-checking in app layer")
        return f"❌ **Error during fact-checking:** {str(e)}"


def use_sample_claim(sample_dropdown: str) -> str:
    """Return the selected sample claim."""
    if sample_dropdown and sample_dropdown != "-- Select a sample claim --":
        return sample_dropdown
    return ""


# ---------------------------------------------------------------------------#
#  Gradio Interface
# ---------------------------------------------------------------------------#

def create_interface():
    """Create and configure the Gradio interface."""

    # Load sample claims
    sample_claims = load_sample_claims()
    sample_options = ["-- Select a sample claim --"] + sample_claims

    # Create Gradio Blocks interface
    with gr.Blocks(title="Fact-Checker") as demo:

        gr.Markdown("# 🔍 Fact-Checker")
        gr.Markdown("*Verify claims against trusted sources using AI-powered analysis*")

        with gr.Tabs():

            # Main Fact-Checking Tab
            with gr.Tab("📝 Fact Check"):

                with gr.Row():
                    with gr.Column(scale=3):
                        claim_input = gr.Textbox(
                            label="Enter claim to verify",
                            placeholder="Example: India's GDP growth rate reached 8.2% in Q1 2024.",
                            lines=4,
                            max_lines=10
                        )

                    with gr.Column(scale=1):
                        sample_dropdown = gr.Dropdown(
                            choices=sample_options,
                            label="📋 Sample Claims",
                            value="-- Select a sample claim --"
                        )
                        use_sample_btn = gr.Button("📝 Use This Sample", size="sm")

                check_btn = gr.Button("🔍 Check Fact", variant="primary", size="lg")

                result_output = gr.Markdown(
                    value="""
> ## 👋 Welcome to Fact-Checker!
>
> Get started by entering a claim or selecting a sample from the dropdown.
>
> ### 🚀 How to Use:
>
> 1. **Enter a claim** in the text box above (or select a sample)
> 2. **Click "Check Fact"** to analyze the claim
> 3. **Review results** with AI-generated verdict, confidence, and evidence

<br>

💡 **Tip:** Make sure the database is initialized in the Configuration tab before checking facts!

Results will appear here after you check your first claim.
"""
                )

                # Button actions
                check_btn.click(
                    fn=check_fact,
                    inputs=[claim_input],
                    outputs=[result_output]
                )

                use_sample_btn.click(
                    fn=use_sample_claim,
                    inputs=[sample_dropdown],
                    outputs=[claim_input]
                )

            # Configuration Tab
            with gr.Tab("⚙️ Configuration"):
                gr.Markdown("""
## 🗄️ Database Initialization

Initialize the verified facts database to enable fact-checking capabilities.
""")

                init_btn = gr.Button(
                    "🔄 Initialize Fact Database",
                    variant="primary",
                    size="lg"
                )

                init_output = gr.Markdown(
                    value="""
> ## 📋 Ready to Initialize
>
> Click the **Initialize Fact Database** button above to:
>
> - 📥 Load verified facts from `tests/verified_facts_db.csv`
> - 🔄 Generate embeddings using Together AI
> - 💾 Create FAISS vector index for semantic search
> - ✅ Enable fact-checking capabilities

<br>

💡 **Note:** This is a one-time setup process (unless you delete the `faiss_index_facts` folder).
"""
                )

                init_btn.click(
                    fn=initialize_database,
                    inputs=[],
                    outputs=[init_output]
                )

                gr.Markdown("<br><br>")

                gr.Markdown("""
## ℹ️ Quick Guide

> ### 📝 Steps to Verify a Claim:
>
> 1. **Initialize** the database using the button above (one-time setup)
> 2. **Navigate** to the "Fact Check" tab
> 3. **Enter** a claim or select a sample from the dropdown
> 4. **Click** "Check Fact" to run the verification
> 5. **Review** the AI-generated verdict with evidence and reasoning

<br>

> ### 🎯 Understanding Results:
>
> **Verdict Types:**
> - ✅ **Likely True** — Evidence supports the claim
> - ❌ **Likely False** — Evidence contradicts the claim
> - 🤷‍♂️ **Unverifiable** — Insufficient or conflicting evidence
>
> **Confidence Indicators:**
> - 🟢 **High** — Strong evidence match with verified facts
> - 🟡 **Medium** — Moderate evidence alignment
> - 🔴 **Low** — Weak or minimal evidence found
""")

            # About Tab
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                # About Fact-Checker

                This AI-powered fact-checking system verifies claims from news and social media against a database of verified facts using Retrieval-Augmented Generation (RAG).

                ## How It Works

                1. **Claim Extraction**: Uses spaCy NLP to extract key claims and entities
                2. **Fact Retrieval**: Searches for similar verified facts using FAISS vector database
                3. **LLM Classification**: Uses Meta-Llama-3.1-8B to classify claims with reasoning
                4. **Structured Output**: Provides verdict, confidence, evidence, and reasoning

                ## Technology Stack

                - **NLP**: spaCy (en_core_web_sm)
                - **Embeddings**: BAAI/bge-base-en-v1.5
                - **Vector DB**: FAISS
                - **LLM**: Meta-Llama-3.1-8B-Instruct-Turbo (via Together AI)
                - **UI**: Gradio

                ## Source Code

                This project demonstrates:
                - Retrieval-Augmented Generation (RAG)
                - Object-Oriented Programming (SOLID principles)
                - Prompt Engineering
                - Production ML System Design
                """)

        gr.Markdown("---")
        gr.Markdown("*Built with ❤️ using Gradio, LangChain, and Together AI*")

    return demo


# ---------------------------------------------------------------------------#
#  Main Entry Point
# ---------------------------------------------------------------------------#

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting Fact-Checker application")
    logger.info("=" * 60)

    # Auto-initialize database on startup (for HuggingFace Spaces deployment)
    if not os.path.exists("faiss_index_facts"):
        print("🔄 Auto-initializing fact database...")
        logger.info("Auto-initializing fact database on startup")
        try:
            status = initialize_database()
            print("✅ Database initialized successfully")
            logger.info("Auto-initialization completed successfully")
        except Exception as e:
            print(f"⚠️ Database initialization failed: {str(e)}")
            logger.error(f"Auto-initialization failed: {str(e)}")
            print("💡 You can manually initialize from the Configuration tab")
    else:
        print("✅ Database already initialized")
        logger.info("Database detected at startup")
        DATABASE_INITIALIZED = True

    logger.info("Creating Gradio interface")
    demo = create_interface()

    logger.info("Launching Gradio app on port 7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
    logger.info("Application shutdown")
