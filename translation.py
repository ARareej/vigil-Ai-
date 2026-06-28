from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

def detect_prompt_language(prompt):
    try:
        lang_code = detect(prompt)
        return lang_code
    except LangDetectException:
        return "unknown"

def translate_prompt_to_english(prompt):
    try:
        translator = GoogleTranslator(source='auto', target='en')
        translated = translator.translate(prompt)
        return translated
    except Exception as e:
        print(f"  Translation failed: {str(e)}")
        return prompt

def prepare_prompt_for_classification(prompt):
    original_prompt = prompt
    detected_language = detect_prompt_language(prompt)
    was_translated = False
    processed_prompt = prompt

    if detected_language == 'ar':
        processed_prompt = translate_prompt_to_english(prompt)
        was_translated = True

    return {
        "original_prompt": original_prompt,
        "processed_prompt": processed_prompt,
        "detected_language": detected_language,
        "was_translated": was_translated
    }

if __name__ == "__main__":
    # Test
    print(prepare_prompt_for_classification("Generate a beautiful landscape"))
    print(prepare_prompt_for_classification("إنشاء منظر طبيعي جميل"))
    print(prepare_prompt_for_classification("Créez un paysage magnifique"))
