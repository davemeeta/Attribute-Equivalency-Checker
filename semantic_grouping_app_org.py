import streamlit as st
import pandas as pd
import re
import json
import ollama

# ----------------- Helper Functions -----------------
def normalize_column(col):
    col = str(col).strip()
    col = re.sub(r'([a-z])([A-Z])', r'\1 \2', col)
    col = col.replace("_", " ").replace("-", " ")
    col = col.lower()
    col = re.sub(r'\s+', ' ', col).strip()
    return col

def build_prompt(columns):
    normalized = [normalize_column(c) for c in columns]

    prompt = f"""
You are an expert database semantic analyzer.

Your job:
Cluster column names ONLY if they represent EXACT SAME concept.

CRITICAL:
- Never mix first name and last name.
- Never mix different entity types.
- Prefer splitting over merging.
- Output STRICT JSON only.
- DO NOT use "..." anywhere. List all attributes fully.
- Output must be a JSON ARRAY even if only one group.

Columns:
{normalized}

Return format (JSON ONLY):
[
  {{
    "attributes": ["col1", "col2"],
    "recommended_name": "some_name",
    "confidence": 0.0
  }}
]
"""
    return prompt

def extract_json_any(content: str) -> str:
    """
    Extract either a JSON array [...] or object {...} from model output.
    Prefer arrays; fallback to objects.
    """
    text = content.strip()

    # If already pure JSON
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
        return text

    # Try find array
    a_start = text.find("[")
    a_end = text.rfind("]")
    if a_start != -1 and a_end != -1 and a_end > a_start:
        return text[a_start:a_end + 1]

    # Try find object
    o_start = text.find("{")
    o_end = text.rfind("}")
    if o_start != -1 and o_end != -1 and o_end > o_start:
        return text[o_start:o_end + 1]

    raise ValueError("No JSON found in model output")

def group_columns(columns, model="llama3"):
    prompt = build_prompt(columns)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response["message"]["content"]
        json_text = extract_json_any(content)

        # Remove illegal ellipsis that breaks JSON (common failure)
        json_text = json_text.replace("...", "")

        parsed = json.loads(json_text)

        # If model returned an object, wrap it into a list
        if isinstance(parsed, dict):
            parsed = [parsed]

        # If model returned something unexpected, fail cleanly
        if not isinstance(parsed, list):
            raise ValueError("Parsed JSON is not a list")

        # Ensure each item is a dict with required keys
        cleaned = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            attrs = item.get("attributes", [])
            if isinstance(attrs, str):
                attrs = [attrs]
            if not isinstance(attrs, list):
                attrs = []

            cleaned.append({
                "attributes": [str(a) for a in attrs],
                "recommended_name": str(item.get("recommended_name", "")),
                "confidence": item.get("confidence", 0.0)
            })

        if not cleaned:
            raise ValueError("No valid groups returned")

        return cleaned

    except Exception as e:
        return {"error": str(e), "raw_output": content if 'content' in locals() else ""}

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="CSV Column Semantic Grouping", layout="wide")

st.title("CSV Column Semantic Grouping (Ollama)")
st.write("Upload a CSV file and group its columns semantically using your local LLaMA model (Ollama).")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

model_name = st.text_input("Ollama model name", value="llama3")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        cols = list(df.columns)

        st.subheader("Detected Columns")
        st.write(cols)

        if st.button("Group Columns Semantically"):
            with st.spinner("Grouping columns..."):
                result = group_columns(cols, model=model_name)

            if isinstance(result, dict) and "error" in result:
                st.error(f"Error: {result['error']}")
                st.text("Raw output from model:")
                st.code(result.get("raw_output", ""))
            else:
                st.success("Columns grouped successfully!")
                st.json(result)

                # Download mapping CSV
                flat_result = []
                for group in result:
                    for attr in group.get("attributes", []):
                        flat_result.append({
                            "original_column": attr,
                            "recommended_name": group.get("recommended_name"),
                            "confidence": group.get("confidence")
                        })

                out_df = pd.DataFrame(flat_result)
                st.download_button(
                    label="Download Grouped Columns CSV",
                    data=out_df.to_csv(index=False),
                    file_name=uploaded_file.name.replace(".csv", "_grouped.csv"),
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
