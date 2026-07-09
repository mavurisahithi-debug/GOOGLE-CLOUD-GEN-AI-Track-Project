from google import genai

# 1. Initialize the latest Gemini Client with your exact API Key
API_KEY = "AQ.Ab8RN6KebRYP_acQgfV7gjPwoapNJlxjQkRTR3_0Gx6hEIb38A"
client = genai.Client(api_key=API_KEY)

# 2. Support Guidance Function
def generate_learning_support(student_text, detected_emotion):
    # Using the updated recommended gemini-2.5-flash model
    model_name = "gemini-2.5-flash"
    
    # Custom system prompt for Epic 4
    prompt = f"""
    You are an empathetic academic virtual assistant. 
    A student has expressed the following challenge: "{student_text}"
    Our emotion engine detected that the student feels: {detected_emotion}.
    
    Provide a highly encouraging, structured support response. 
    1. Acknowledge their feeling gently without explicitly saying "Our engine detected you are X".
    2. Provide 2-3 actionable steps or tips to resolve their academic challenge.
    3. End with an inspiring, supportive sign-off.
    """
    
    print("\nContacting the updated Gemini AI Engine for personalized support...")
    
    # Using the latest client.models.generate_content API format
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return response.text

# Test the engine independently
if __name__ == "__main__":
    sample_text = "This code error is driving me crazy, I can't solve it"
    sample_emotion = "Frustrated"
    
    ai_guidance = generate_learning_support(sample_text, sample_emotion)
    print("\n=== Gemini AI Generated Support ===")
    print(ai_guidance)