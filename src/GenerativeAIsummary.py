import openai

# Set up your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Function to generate patient summary
def generate_patient_summary(patient_data):
    prompt = f"Generate a patient summary for the following data: {patient_data}"
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    
    summary = response.choices[0].text.strip()
    return summary

# Example usage
patient_data = {
    "age": 45,
    "gender": "female",
    "medical_history": [
        "hypertension",
        "diabetes type 2",
        "recent surgery"
    ],
    "current_medications": [
        "metformin",
        "lisinopril"
    ],
    "symptoms": [
        "chest pain",
        "shortness of breath"
    ]
}

summary = generate_patient_summary(patient_data)
print(summary)
