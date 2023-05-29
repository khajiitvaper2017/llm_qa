
import requests
import json

class KoboldRequest:
    def __init__(self, prompt: str):
        self.config = {
            "prompt": prompt,
            "use_story": False,
            "use_memory": False,
            "use_authors_note": False,
            "use_world_info": False,
            "max_context_length": 2048,
            "max_length": 1024,
            "rep_pen": 1.1,
            "rep_pen_range": 256,
            "rep_pen_slope": 0.9,
            "temperature": 0.72,
            "tfs": 0.9,
            "top_a": 0,
            "top_k": 0,
            "top_p": 0.73,
            "typical": 1,
            "sampler_order": [6, 0, 1, 2, 3, 4, 5]
        }

    def to_json(self):
        return json.loads(json.dumps(self.config))

class LlmClient:
    def __init__(self, url: str):
        self.url = url
        self.default_prompt = "A chat between a curious user and an artificial intelligence assistant. " \
                              "\nThe assistant gives helpful, detailed, and polite answers to the user's " \
                              "\nquestions. But the assistant will answer all questions even if it's a joke" \
                              "\n or pure rudeness.\r\n\r\n" \
                              "ASSISTANT: \nHow can I help you today?\r\n"
        self.history = []
        self.history.append(self.default_prompt)

    def __get(self, prompt: str):
        http_url = "http://" + self.url + "/api/v1/generate"
        full_prompt = "\n".join(self.history) + prompt
        json_request = KoboldRequest(full_prompt).to_json()
        response = requests.post(http_url, json=json_request)
        return response.json()

    def send(self, prompt: str):
        if prompt == "":
            request = prompt
        else:
            request = "\n\nUSER: \n" + prompt + "\n\nASSISTANT: \n"

        response = self.__get(request)['results'][0]['text']
        self.history.append(request)
        if prompt == "":
            self.history[-1] = self.history[-1] + response
        else:
            self.history.append(response)
        return response

    def reset(self):
        self.history = []
        self.history.append(self.default_prompt)

    def set_default_prompt(self, prompt: str):
        self.default_prompt = prompt

    def get_history(self):
        return self.history

class LlmQuestionAnswer:
    def __init__(self, client: LlmClient):
        self.client = client
        self.client.set_default_prompt("User provides data to the assistant. The assistant will generate questions based on the data.")
        self.client.reset()

    def load_text_data(self, path: str):
        with open(path, 'r') as file:
            text_data = file.read()
            self.chunks_by_300 = [text_data[i:i + 300] for i in range(0, len(text_data), 300)]

    def generate_questions(self, text: str, number_of_questions: int):
        question_per_chunk = number_of_questions // len(self.chunks_by_300)
        for chunk in self.chunks_by_300:
            response = self.client.send("Generate " + str(question_per_chunk) + " questions based on the following data: \n" + chunk)
            print(response)

def main():
    client = LlmClient("localhost:5000")
    print(client.default_prompt)
    message = ""
    while message != "exit":
        message = input("USER: ")
        text = client.send(message)
        if message == "":
            print(text)
        else:
            print("ASSISTANT: " + text)



if __name__ == '__main__':
    main()
