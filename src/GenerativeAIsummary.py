
#Generative AI Application
#Utilize OpenAI's GPT-3.5 to generate patient summaries:
import openai

# Set up your OpenAI API key

openai.api_key = 'your_key'

# Function to generate patient summary
def generate_patient_summary(patient_data):
    prompt = f"Generate a patient summary for the following data: {patient_data}"
    

    # Replace 'text-davinci-003' with 'gpt-3.5-turbo-instruct'
    
    response = openai.ChatCompletion.create( 
    model="gpt-3.5-turbo-instruct",
    prompt="Your prompt here",
    max_tokens=50
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
