from flask import Flask, request, jsonify
import openai

app = Flask(__name__)

# Set your OpenAI API key here
import os
openai.api_key = os.getenv("OPENAI_API_KEY")


# Synonym mapping for research types
RESEARCH_TYPE_SYNONYMS = {
    "A/B (Split Test)": ["experiment", "tests", "test", "testing", "A/B", "AB test"],
    "User Study": ["user research", "usability study", "usability research"],
    "Market Research": ["customer research", "customer interview", "stakeholder interview"],
    "Lead generation": ["drove leads"],
    "Copy": ["messaging framework", "copy", "messaging"],
    "Ecommerce": [
        "orders per visitor",
        "orders",
        "online orders",
        "average order value",
        "revenue per visitor",
        "online purchases",
        "purchases",
        "discounted",
    ],
    "Data Analysis": [
        "gradually increasing",
        "gradually decreasing",
        "order trend",
        "purchase trend",
        "purchasing trend",
    ],
}

def get_keywords_from_ai(text, keyword_list, synonyms=None):
    """
    Use AI to process and extract keywords from text.
    """
    prompt = f"""
    You are an intelligent assistant. Given the text: "{text}", and the following keyword list: {keyword_list},
    extract the keywords that are relevant to the text. If synonyms are provided for a keyword, consider them as well.
    Synonyms: {synonyms if synonyms else "None"}
    Return the keywords as a list.
    """
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
        )
        # Extract and return the response as a list
        keywords = response.choices[0].text.strip().split(",")
        return [kw.strip() for kw in keywords if kw.strip()]
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return []

@app.route('/process_insight', methods=['POST'])
def process_insight():
    try:
        data = request.get_json()
        summary = data.get("summary", "")  # Default to empty string if not provided

        # Use the predefined values if they are not passed in the request
        goals = data.get("goals", [])
        categories = data.get("categories", [])
        tools = data.get("tools", [])
        elements = data.get("elements", [])
        research_types = data.get("research_types", [])
        industries = data.get("industries", [])

        if not summary:
            return jsonify({"error": "Summary text is required"}), 400

        # Call AI-powered keyword extraction
        selected_categories = get_keywords_from_ai(summary, categories, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_elements = get_keywords_from_ai(summary, elements, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_tools = get_keywords_from_ai(summary, tools)
        selected_goals = get_keywords_from_ai(summary, goals, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_research_types = get_keywords_from_ai(summary, research_types, synonyms=RESEARCH_TYPE_SYNONYMS)
        selected_industries = get_keywords_from_ai(summary, industries)

        # Return the auto-selected values
        return jsonify(
            {
                "selected_categories": selected_categories,
                "selected_elements": selected_elements,
                "selected_tools": selected_tools,
                "selected_goals": selected_goals,
                "selected_research_types": selected_research_types,
                "selected_industries": selected_industries,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
