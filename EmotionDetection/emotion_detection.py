"""Runs emotion detection over text using the Watson NLP EmotionPredict service."""
import json
import requests

EMOTION_KEYS = ('anger', 'disgust', 'fear', 'joy', 'sadness')


def _blank_result():
    result = {key: None for key in EMOTION_KEYS}
    result['dominant_emotion'] = None
    return result


def emotion_detector(text_to_analyze):
    """Return emotion scores and the dominant emotion for the given text."""
    if not text_to_analyze or text_to_analyze.strip() == "":
        return _blank_result()

    url = ('https://sn-watson-emotion.labs.skills.network/v1/watson.runtime'
           '.nlp.v1/NlpService/EmotionPredict')
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    myobj = {"raw_document": {"text": text_to_analyze}}

    response = requests.post(url, json=myobj, headers=header, timeout=10)

    if response.status_code == 400:
        return _blank_result()

    try:
        formatted_response = json.loads(response.text)
        emotions = formatted_response['emotionPredictions'][0]['emotion']

        result = {key: emotions[key] for key in EMOTION_KEYS}
        result['dominant_emotion'] = max(emotions, key=emotions.get)
        return result
    except (KeyError, IndexError, json.JSONDecodeError):
        return _blank_result()
